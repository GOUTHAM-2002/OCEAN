"""Eval worker for the last-dance adapters: load WizardLM-13B once, hot-swap each
assigned person's adapter, score OCEAN on BFI-2 + FFPI (10 seeds), write
results_lastdance/<cid>.json. Usage: python lastdance_eval.py <gpu> <cids_csv>
"""
import os, sys, re, json
os.environ["HF_HOME"] = "/mnt/ssd3/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from math import sqrt
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

GPU = sys.argv[1]; CIDS = sys.argv[2].split(",")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
DEV = "cuda:0"; MODEL = "WizardLMTeam/WizardLM-13B-V1.2"
ADAP = "adapters_lastdance"; OUTD = "results_lastdance"; os.makedirs(OUTD, exist_ok=True)
N_SEEDS = 10; BATCH = 10; TCI = 2.262  # t_.975 df=9
CODES = ["O", "C", "E", "A", "N"]
ACC = ("1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
       "4 = Moderately Accurate\n5 = Very Accurate")
AGREE = ("1 = Disagree strongly\n2 = Disagree a little\n3 = Neutral; no opinion\n"
         "4 = Agree a little\n5 = Agree strongly")
CONFIGS = [dict(q="bfi2_questions.csv", title="BFI-2",
                instr=f"Indicate how much you agree that each statement describes you as you generally are now, using this scale:\n{AGREE}", stem="I am someone who..."),
           dict(q="ffpi_questions.csv", title="FFPI",
                instr=f"Indicate how accurately each short statement describes you as you generally are now, using this scale:\n{ACC}", stem="Statements:")]
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
def vic(u): return ("A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed answers to the user's questions. "
                    f"USER: {SYSTEM}\n\n{u} ASSISTANT:")
Q = {}
for cfg in CONFIGS:
    q = pd.read_csv(cfg["q"]).sort_values("item_number").reset_index(drop=True)
    n = len(q); items = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
    user = (f"{cfg['instr']}\n\n{cfg['stem']}\n{items}\n\nRespond with ONLY a JSON object mapping EVERY "
            f'item number to its 1-5 rating, like {{"1": 4, ... , "{n}": 2}}. Output nothing but the JSON.')
    Q[cfg["title"]] = dict(prompt=vic(user), n=n, rev=(q["scoring_direction"] == "reverse").values, dom=q["domain_code"].values)

WORD2NUM = {"very inaccurate":1,"moderately inaccurate":2,"neither accurate nor inaccurate":3,"neither":3,"moderately accurate":4,"very accurate":5,"disagree strongly":1,"strongly disagree":1,"disagree a little":2,"disagree somewhat":2,"disagree":2,"neutral":3,"no opinion":3,"neutral; no opinion":3,"neither agree nor disagree":3,"agree a little":4,"agree somewhat":4,"agree":4,"agree strongly":5,"strongly agree":5}
def _val(v):
    if isinstance(v,(int,float)): iv=int(v); return iv if 1<=iv<=5 else None
    if isinstance(v,str):
        s=v.strip().lower().rstrip(".")
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

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": DEV})
if getattr(base.config, "pretraining_tp", 1) not in (1, None): base.config.pretraining_tp = 1
base.eval()
print(f"[GPU{GPU}] base loaded ({len(CIDS)} adapters)", flush=True)

@torch.no_grad()
def gen(model, prompt, k, seed, n):
    torch.manual_seed(seed)
    inp = tok([prompt]*k, return_tensors="pt", padding=True).to(DEV)
    out = model.generate(**inp, max_new_tokens=max(500, n*8), do_sample=True, temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
    pl = inp["input_ids"].shape[1]
    return [tok.decode(out[i][pl:], skip_special_tokens=True) for i in range(k)]

def eval_instrument(model, title):
    info = Q[title]; n = info["n"]; runs = []; seed = 0
    while len(runs) < N_SEEDS and seed < 4:
        for t in gen(model, info["prompt"], BATCH, seed, n):
            if len(runs) >= N_SEEDS: break
            arr, miss = parse(t, n)
            if miss <= max(3, int(0.1*n)): runs.append(arr)
        seed += 1
    while len(runs) < N_SEEDS: runs.append([3]*n)
    rev, dom = info["rev"], info["dom"]; st = {}
    for c in CODES:
        xs = np.array([float(np.where(rev,6-np.array(r,float),np.array(r,float))[dom==c].mean()) for r in runs])
        st[c] = [round(xs.mean(),3), round(TCI*xs.std(ddof=1)/sqrt(N_SEEDS),3)]
    return st

pmodel = None
for cid in CIDS:
    outp = os.path.join(OUTD, f"{cid}.json")
    if os.path.exists(outp): print(f"[{cid}] skip", flush=True); continue
    path = os.path.join(ADAP, str(cid))
    name = "a_" + str(cid)
    if pmodel is None: pmodel = PeftModel.from_pretrained(base, path, adapter_name=name)
    else: pmodel.load_adapter(path, adapter_name=name)
    pmodel.set_adapter(name)
    table = {cfg["title"]: eval_instrument(pmodel, cfg["title"]) for cfg in CONFIGS}
    json.dump({"case_id": cid, "table": table}, open(outp, "w"))
    ocean = {c: round(np.mean([table[t][c][0] for t in ["BFI-2","FFPI"]]),2) for c in CODES}
    print(f"[{cid}] done {ocean}", flush=True)
print(f"[GPU{GPU}] LD_EVAL_DONE", flush=True)
