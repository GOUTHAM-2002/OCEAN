"""Exp 1b: causally validate the trait directions by activation steering.
Register a forward hook on one decoder layer that adds c*unit_direction to the
residual stream, then run the BFI-2 + FFPI questionnaire and measure the trait
score. Sweep layers x coefficients. A monotonic score-vs-coefficient curve =
the trait is causally carried by that direction.

Usage: python steer_validate.py <trait> <device> [layers_csv] [coeffs_csv]
"""
import os, sys, re, json
import numpy as np, pandas as pd, torch
from math import sqrt
from mi_common import load_model, vicuna, TRAITS, TRAIT_NAME

TRAIT = sys.argv[1] if len(sys.argv) > 1 else "A"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
LAYERS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["14", "20", "26"])]
COEFFS = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["-0.12", "-0.06", "0", "0.06", "0.12"])]
N_SEEDS = 4   # enough to see the causal trend fast

model, tok = load_model(DEV)
blob = torch.load("steer_dirs.pt")
DIRS, RESID = blob["dirs"], blob["resid_norm_ref"]
print(f"loaded dirs; steering trait {TRAIT} ({TRAIT_NAME[TRAIT]})", flush=True)

# --- questionnaire setup (BFI-2 + FFPI), same scoring as reeval20 ---
ACC = ("1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
       "4 = Moderately Accurate\n5 = Very Accurate")
AGREE = ("1 = Disagree strongly\n2 = Disagree a little\n3 = Neutral; no opinion\n"
         "4 = Agree a little\n5 = Agree strongly")
CONFIGS = [   # BFI-2 only for speed (60 items); causal trend is relative
    dict(q="bfi2_questions.csv", title="BFI-2",
         instr=f"Indicate how much you agree that each statement describes you as you generally are now, using this scale:\n{AGREE}", stem="I am someone who..."),
]
WORD2NUM = {"very inaccurate":1,"moderately inaccurate":2,"neither accurate nor inaccurate":3,"neither":3,
    "moderately accurate":4,"very accurate":5,"disagree strongly":1,"strongly disagree":1,"disagree a little":2,
    "disagree somewhat":2,"disagree":2,"neutral":3,"no opinion":3,"neutral; no opinion":3,"neither agree nor disagree":3,
    "agree a little":4,"agree somewhat":4,"agree":4,"agree strongly":5,"strongly agree":5}
def _val(v):
    if isinstance(v,(int,float)): iv=int(v); return iv if 1<=iv<=5 else None
    if isinstance(v,str):
        s=v.strip().lower().rstrip(".");
        if s.isdigit() and 1<=int(s)<=5: return int(s)
        return WORD2NUM.get(s)
    return None
def parse(text,n):
    d={}; m=re.search(r"\{.*\}",text,re.S)
    if m:
        try: d={str(k):v for k,v in json.loads(m.group(0)).items()}
        except Exception: d={}
    if not d:
        d=dict(re.findall(r'"?(\d{1,3})"?\s*:\s*([1-5])',text))
        if not d: d={k:v for k,v in re.findall(r'"?(\d{1,3})"?\s*:\s*"?([A-Za-z ]+?)"?[,}\n]',text)}
    arr,miss=[],0
    for i in range(1,n+1):
        v=_val(d.get(str(i)))
        if v is None: v=3; miss+=1
        arr.append(v)
    return arr,miss

Q = {}
for cfg in CONFIGS:
    q = pd.read_csv(cfg["q"]).sort_values("item_number").reset_index(drop=True)
    n = len(q); items = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
    user = (f"{cfg['instr']}\n\n{cfg['stem']}\n{items}\n\nRespond with ONLY a JSON object mapping EVERY "
            f'item number to its 1-5 rating, like {{"1": 4, "2": 3, ... , "{n}": 2}}. '
            f"Include all {n} item numbers. Output nothing but the JSON.")
    Q[cfg["title"]] = dict(prompt=vicuna(user), n=n,
                           rev=(q["scoring_direction"]=="reverse").values, dom=q["domain_code"].values)

# --- steering hook (layer-aware: only adds at STEER["layer"]) ---
STEER = {"vec": None, "layer": None}
def make_hook(i):
    def hook(module, inp, out):
        if STEER["vec"] is None or STEER["layer"] != i: return out
        if isinstance(out, tuple):
            return (out[0] + STEER["vec"].to(out[0].dtype),) + out[1:]
        return out + STEER["vec"].to(out.dtype)
    return hook
handles = [model.model.layers[i].register_forward_hook(make_hook(i))
           for i in range(len(model.model.layers))]

@torch.no_grad()
def gen(prompt, k, seed, n):
    torch.manual_seed(seed)
    inp = tok([prompt]*k, return_tensors="pt", padding=True).to(DEV)
    out = model.generate(**inp, max_new_tokens=max(500, n*8), do_sample=True,
                         temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
    pl = inp["input_ids"].shape[1]
    return [tok.decode(out[i][pl:], skip_special_tokens=True) for i in range(k)]

def trait_score():
    """mean of BFI-2+FFPI for TRAIT, current STEER setting."""
    vals = []
    for cfg in CONFIGS:
        info = Q[cfg["title"]]; n = info["n"]; runs = []
        seed = 0
        while len(runs) < N_SEEDS and seed < 6:
            for t in gen(info["prompt"], 4, seed, n):
                if len(runs) >= N_SEEDS: break
                arr, miss = parse(t, n)
                if miss <= max(3, int(0.1*n)): runs.append(arr)
            seed += 1
        while len(runs) < N_SEEDS: runs.append([3]*n)
        rev, dom = info["rev"], info["dom"]
        sc = [float(np.where(rev,6-np.array(r,float),np.array(r,float))[dom==TRAIT].mean()) for r in runs]
        vals.append(np.mean(sc))
    return float(np.mean(vals))

# baseline (no steering)
STEER["vec"] = None
base = trait_score()
print(f"\n=== {TRAIT_NAME[TRAIT]} | baseline (no steer) = {base:.3f} ===", flush=True)

results = {"trait": TRAIT, "baseline": base, "grid": []}
for L in LAYERS:
    d = DIRS[TRAIT][L+1]                       # hidden_states[L+1] = output of decoder layer L
    unit = (d / (d.norm() + 1e-6)).to(DEV)
    scale = float(RESID[L+1])                  # typical residual norm at that layer
    STEER["layer"] = L
    row = []
    for c in COEFFS:
        STEER["vec"] = (unit * c * scale) if c != 0 else None
        s = base if c == 0 else trait_score()
        row.append((c, s))
        print(f"  layer {L:2d}  c={c:+.3f}  {TRAIT_NAME[TRAIT][:4]}={s:.3f}  (Δ {s-base:+.3f})", flush=True)
    results["grid"].append({"layer": L, "curve": row})
    STEER["vec"] = None

json.dump(results, open(f"steer_result_{TRAIT}.json", "w"), indent=2)
print(f"\nsaved steer_result_{TRAIT}.json", flush=True)
