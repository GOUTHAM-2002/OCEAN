"""
Generic runner: a local MoE model (Qwen2-57B-A14B-Instruct, 4-bit, all GPUs)
takes a given OCEAN inventory 5 times (stateless, same prompt, 5 seeds) and
plots the 5 domain means (1-5) with 95% CIs. Same protocol as the IPIP/BFI-2 runs.

Usage: python run_inventory_ocean.py <config_key>
"""
import os, sys
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re, json, csv
from math import sqrt
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"
N_SEEDS = 5
HERE = os.path.dirname(os.path.abspath(__file__))

# instrument configs -------------------------------------------------------
CONFIGS = {
 "goldberg100": dict(
    questions="goldberg100_questions.csv", out="goldberg100",
    title="Goldberg-100 Unipolar Markers",
    instr=("Here are 100 common words. Each describes a personality trait. Please indicate "
           "how accurately each word describes you as you generally are now, using this scale:\n"
           "1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
           "4 = Moderately Accurate\n5 = Very Accurate"),
    stem="Words:"),
 "ffpi": dict(
    questions="ffpi_questions.csv", out="ffpi",
    title="FFPI (Five-Factor Personality Inventory)",
    instr=("Here are 100 short statements describing how a person might behave. For each, "
           "indicate how accurately it describes you as you generally are now, using this scale:\n"
           "1 = Very Inaccurate\n2 = Moderately Inaccurate\n3 = Neither Accurate Nor Inaccurate\n"
           "4 = Moderately Accurate\n5 = Very Accurate"),
    stem="Statements (rate how accurately each describes you):"),
}
cfg = CONFIGS[sys.argv[1]]

q = pd.read_csv(os.path.join(HERE, cfg["questions"])).sort_values("item_number").reset_index(drop=True)
N_ITEMS = len(q)
CODE_NAME = {"O":"Openness","C":"Conscientiousness","E":"Extraversion","A":"Agreeableness","N":"Neuroticism"}

item_lines = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."
USER = (f"{cfg['instr']}\n\n{cfg['stem']}\n{item_lines}\n\n"
        f'Respond with ONLY a JSON object mapping EVERY item number to its 1-5 rating, like '
        f'{{"1": 4, "2": 3, "3": 5, ... , "{N_ITEMS}": 2}}. Include all {N_ITEMS} item numbers '
        f"(1 through {N_ITEMS}). Output nothing but the JSON.")

print("Loading", MODEL_ID, "...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
max_memory = {}
for i in range(torch.cuda.device_count()):
    free_gib = torch.cuda.mem_get_info(i)[0]/(1024**3)
    if free_gib >= 8: max_memory[i] = f"{int(free_gib-4)}GiB"
print("per-GPU budget:", max_memory, flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                             device_map="auto", max_memory=max_memory)
model.eval(); print("Loaded.", flush=True)

prompt_text = tok.apply_chat_template(
    [{"role":"system","content":SYSTEM},{"role":"user","content":USER}],
    tokenize=False, add_generation_prompt=True)

def parse_strict(text):
    """Parse a numbered JSON object {"1":r1,...}. Build the N-length array by item
    number so omissions stay positionally correct; fill rare missing items with 3
    (neutral). Return None (-> retry) only if too many items are missing."""
    d = {}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            raw = json.loads(m.group(0))
            d = {str(k): v for k, v in raw.items()}
        except Exception:
            d = {}
    if not d:  # fallback: pull "num": val pairs directly
        d = dict(re.findall(r'"?(\d{1,3})"?\s*:\s*([1-5])', text))
    arr, miss = [], 0
    for i in range(1, N_ITEMS + 1):
        v = d.get(str(i))
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = None
        if v is None or not (1 <= v <= 5):
            v = 3; miss += 1
        arr.append(v)
    if miss > max(3, int(0.1 * N_ITEMS)):
        return None
    return arr

def gen_batch(prompts, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    inp = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max(800, N_ITEMS*10), do_sample=True,
                             temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
    plen=inp["input_ids"].shape[1]
    return [tok.decode(out[i][plen:], skip_special_tokens=True) for i in range(len(prompts))]

def take_all(n):
    results=[None]*n
    for i,t in enumerate(gen_batch([prompt_text]*n, 0)):
        results[i]=parse_strict(t)
    tries=0
    while any(r is None for r in results) and tries<10:
        tries+=1
        idxs=[i for i,r in enumerate(results) if r is None]
        print(f"  retry {tries}: regenerating {len(idxs)} run(s) that returned !={N_ITEMS} answers", flush=True)
        for j,t in enumerate(gen_batch([prompt_text]*len(idxs), 100+tries)):
            results[idxs[j]]=parse_strict(t)
    if any(r is None for r in results):
        raise ValueError("could not obtain exactly N answers for all runs after retries")
    return results

rev=(q["scoring_direction"]=="reverse").values; dom=q["domain_code"].values
def domain_scores(raw):
    rc=np.where(rev,6-np.array(raw,float),np.array(raw,float))
    return {c: float(rc[dom==c].mean()) for c in CODE_NAME}

print(f"Running {N_SEEDS} stateless test-takings ...", flush=True)
raw_log=take_all(N_SEEDS); all_runs=[]
for s,raw in enumerate(raw_log):
    sc=domain_scores(raw); all_runs.append(sc)
    print(f"  run {s}:", {k:round(v,2) for k,v in sc.items()}, flush=True)

with open(os.path.join(HERE,f"{cfg['out']}_qwen_raw_answers.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["seed"]+[f"item{i+1}" for i in range(N_ITEMS)])
    for s,raw in enumerate(raw_log): w.writerow([s]+raw)

codes=["O","C","E","A","N"]; means=[]; ci=[]; T=2.776
print()
for c in codes:
    xs=np.array([r[c] for r in all_runs]); m=xs.mean(); se=xs.std(ddof=1)/sqrt(len(xs))
    means.append(m); ci.append(T*se)
    print(f"{CODE_NAME[c]:18s} mean={m:.2f}  95%CI=+/-{T*se:.2f}", flush=True)

fig,ax=plt.subplots(figsize=(8,5.5)); x=np.arange(5)
ax.bar(x,means,yerr=ci,capsize=8,color=["#4C72B0","#55A868","#C44E52","#8172B3","#CCB974"],
       edgecolor="black",alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([CODE_NAME[c] for c in codes],rotation=15)
ax.set_ylabel("Score (1 = low, 5 = high)"); ax.set_ylim(1,5)
ax.set_title(f"Qwen2-57B-A14B-Instruct (MoE) on {cfg['title']} — OCEAN\n"
             f"(mean of {N_SEEDS} stateless runs, error bars = 95% CI)")
ax.grid(axis="y",alpha=0.3)
for xi,m in zip(x,means): ax.text(xi,m+0.05,f"{m:.2f}",ha="center",va="bottom",fontsize=9)
plt.tight_layout()
out=os.path.join(HERE,f"{cfg['out']}_ocean_barchart.png"); plt.savefig(out,dpi=150)
print("Saved plot ->", out, flush=True)
