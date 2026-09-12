"""
Run the same 4 OCEAN tests (IPIP-300, BFI-2, Goldberg-100, FFPI), 5 stateless
seeds each, on an SFT-ONLY (no DPO) open MoE model:
    NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT  (Mixtral 8x7B, 47B, Apache-2.0)
4-bit, spread across all currently-free GPUs. Prints the combined OCEAN table.
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re, json, csv
from math import sqrt
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import subprocess

MODEL_ID = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"
TAG = "sft_mixtral"
N_SEEDS = 5
HERE = os.path.dirname(os.path.abspath(__file__))

ACC = ("1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
       "4 = Moderately Accurate\n5 = Very Accurate")
AGREE = ("1 = Disagree strongly\n2 = Disagree a little\n3 = Neutral; no opinion\n"
         "4 = Agree a little\n5 = Agree strongly")
CONFIGS = [
 dict(key="ipip300", questions="ipip300_questions.csv", title="IPIP-NEO-300",
      instr=("Indicate how accurately each statement describes you as you generally are now, "
             f"using this scale:\n{ACC}"), stem="Statements:"),
 dict(key="bfi2", questions="bfi2_questions.csv", title="BFI-2",
      instr=("Indicate how much you agree that each statement describes you as you generally "
             f"are now, using this scale:\n{AGREE}"), stem="I am someone who..."),
 dict(key="goldberg100", questions="goldberg100_questions.csv", title="Goldberg-100",
      instr=("Each word describes a personality trait. Indicate how accurately each word "
             f"describes you as you generally are now, using this scale:\n{ACC}"), stem="Words:"),
 dict(key="ffpi", questions="ffpi_questions.csv", title="FFPI",
      instr=("Indicate how accurately each short statement describes you as you generally are "
             f"now, using this scale:\n{ACC}"), stem="Statements:"),
]
CODE_NAME = {"O":"Openness","C":"Conscientiousness","E":"Extraversion","A":"Agreeableness","N":"Neuroticism"}
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."

# ---- load model across currently-FREE gpus ----
print("Loading", MODEL_ID, "...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

# pick GPUs that are idle right now (low util AND enough free mem) -> "all free gpus"
q = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"]).decode().strip().splitlines()
max_memory = {}
for line in q:
    idx, free_mb, util = [int(x) for x in line.split(",")]
    if util < 50 and free_mb > 12000:
        max_memory[idx] = f"{int(free_mb/1024)-4}GiB"
assert max_memory, "no free GPUs right now"
print("using free GPUs / budget:", max_memory, flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                             device_map="auto", max_memory=max_memory)
model.eval(); print("Loaded.", flush=True)


def build_prompt(cfg, q):
    item_lines = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
    n = len(q)
    user = (f"{cfg['instr']}\n\n{cfg['stem']}\n{item_lines}\n\n"
            f'Respond with ONLY a JSON object mapping EVERY item number to its 1-5 rating, like '
            f'{{"1": 4, "2": 3, ... , "{n}": 2}}. Include all {n} item numbers. Output nothing but the JSON.')
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                                   tokenize=False, add_generation_prompt=True)

def parse_strict(text, n):
    d = {}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try: d = {str(k): v for k, v in json.loads(m.group(0)).items()}
        except Exception: d = {}
    if not d:
        d = dict(re.findall(r'"?(\d{1,3})"?\s*:\s*([1-5])', text))
    arr, miss = [], 0
    for i in range(1, n+1):
        v = d.get(str(i))
        try: v = int(v)
        except (TypeError, ValueError): v = None
        if v is None or not (1 <= v <= 5): v = 3; miss += 1
        arr.append(v)
    return arr if miss <= max(3, int(0.1*n)) else None

def gen_batch(prompt, k, seed, n):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    inp = tok([prompt]*k, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max(800, n*12), do_sample=True,
                             temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
    plen = inp["input_ids"].shape[1]
    return [tok.decode(out[i][plen:], skip_special_tokens=True) for i in range(k)]

def take_all(prompt, n):
    res = [parse_strict(t, n) for t in gen_batch(prompt, N_SEEDS, 0, n)]
    tries = 0
    while any(r is None for r in res) and tries < 10:
        tries += 1
        idxs = [i for i,r in enumerate(res) if r is None]
        print(f"    retry {tries}: {len(idxs)} run(s)", flush=True)
        for j,t in enumerate(gen_batch(prompt, len(idxs), 100+tries, n)):
            res[idxs[j]] = parse_strict(t, n)
    if any(r is None for r in res): raise ValueError("unparseable runs remain")
    return res

summary = {}
for cfg in CONFIGS:
    qdf = pd.read_csv(os.path.join(HERE, cfg["questions"])).sort_values("item_number").reset_index(drop=True)
    n = len(qdf); rev = (qdf["scoring_direction"]=="reverse").values; dom = qdf["domain_code"].values
    print(f"\n=== {cfg['title']} ({n} items) ===", flush=True)
    prompt = build_prompt(cfg, qdf)
    runs = take_all(prompt, n)
    with open(os.path.join(HERE, f"{TAG}_{cfg['key']}_raw_answers.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed"]+[f"item{i+1}" for i in range(n)])
        for s,r in enumerate(runs): w.writerow([s]+r)
    per = {c: [] for c in CODE_NAME}
    for r in runs:
        rc = np.where(rev, 6-np.array(r,float), np.array(r,float))
        for c in CODE_NAME: per[c].append(float(rc[dom==c].mean()))
    stats = {}
    for c in CODE_NAME:
        xs = np.array(per[c]); m = xs.mean(); ci = 2.776*xs.std(ddof=1)/sqrt(N_SEEDS)
        stats[c] = (m, ci); print(f"  {CODE_NAME[c]:18s} {m:.2f} +/-{ci:.2f}", flush=True)
    summary[cfg["title"]] = stats
    # plot
    codes = ["O","C","E","A","N"]
    fig, ax = plt.subplots(figsize=(8,5.5)); x = np.arange(5)
    ax.bar(x, [stats[c][0] for c in codes], yerr=[stats[c][1] for c in codes], capsize=8,
           color=["#4C72B0","#55A868","#C44E52","#8172B3","#CCB974"], edgecolor="black", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([CODE_NAME[c] for c in codes], rotation=15)
    ax.set_ylabel("Score (1 = low, 5 = high)"); ax.set_ylim(1,5)
    ax.set_title(f"Nous-Hermes-2-Mixtral-8x7B-SFT (no DPO) on {cfg['title']} — OCEAN\n"
                 f"(mean of {N_SEEDS} stateless runs, 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    for xi,c in zip(x,codes): ax.text(xi, stats[c][0]+0.05, f"{stats[c][0]:.2f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, f"{TAG}_{cfg['key']}_barchart.png"), dpi=150)

# ---- combined table ----
print("\n================ Nous-Hermes-2-Mixtral-8x7B-SFT (no DPO) — OCEAN table ================", flush=True)
order = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
code = {v:k for k,v in CODE_NAME.items()}
hdr = f"{'Domain':18s} " + " ".join(f"{cfg['title']:>16s}" for cfg in CONFIGS)
print(hdr)
for d in order:
    row = f"{d:18s} " + " ".join(f"{summary[cfg['title']][code[d]][0]:6.2f} +/-{summary[cfg['title']][code[d]][1]:4.2f}  "
                                  for cfg in CONFIGS)
    print(row)
print("DONE_ALL", flush=True)
