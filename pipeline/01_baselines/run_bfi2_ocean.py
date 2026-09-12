"""
Same experiment as the IPIP-NEO-300 run, but on the BFI-2 (60 items) — a fully
different questionnaire that measures the same OCEAN traits.

The local MoE model (Qwen2-57B-A14B-Instruct, 4-bit) takes the BFI-2 5 times.
Each run is a fresh, stateless request with the SAME exact prompt (no history);
only the RNG seed changes. Output: a bar chart of the 5 OCEAN scores (y 1-5)
with 95% confidence intervals.
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re, json, csv
from math import sqrt
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"
N_SEEDS = 5
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- load the 60 BFI-2 questions ----
q = pd.read_csv(os.path.join(HERE, "bfi2_questions.csv")).sort_values("item_number").reset_index(drop=True)
N_ITEMS = len(q)
assert N_ITEMS == 60
# standard OCEAN labels for the plot (BFI-2 calls two of them differently)
CODE_NAME = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
             "A": "Agreeableness", "N": "Neuroticism"}

# ---- single fixed prompt (identical every run) ----
item_lines = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."
USER = f"""Here are a number of characteristics that may or may not apply to you. Please indicate \
how much you agree or disagree that each statement describes you. Describe yourself as you \
generally are now. Use this 1-5 scale:

1 = Disagree strongly
2 = Disagree a little
3 = Neutral; no opinion
4 = Agree a little
5 = Agree strongly

I am someone who...
{item_lines}

Respond with ONLY a JSON object of the form {{"answers": [a1, a2, ..., a{N_ITEMS}]}} where each \
a_i is an integer from 1 to 5 giving your rating for statement i. The list must have exactly \
{N_ITEMS} integers, in order. Output nothing but the JSON."""

# ---- load model (4-bit, dynamic per-GPU budget on this shared box) ----
print("Loading", MODEL_ID, "...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
n_gpu = torch.cuda.device_count()
max_memory = {}
for i in range(n_gpu):
    free_gib = torch.cuda.mem_get_info(i)[0] / (1024**3)
    if free_gib >= 8:
        max_memory[i] = f"{int(free_gib - 4)}GiB"
print("per-GPU budget:", max_memory, flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                             device_map="auto", max_memory=max_memory)
model.eval()
print("Loaded.", flush=True)

prompt_text = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    tokenize=False, add_generation_prompt=True)


def parse_answers(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            arr = [int(x) for x in json.loads(m.group(0)).get("answers", []) if 1 <= int(x) <= 5]
            if len(arr) == N_ITEMS:
                return arr
        except Exception:
            pass
    digits = [int(d) for d in re.findall(r"[1-5]", text)]
    if len(digits) >= N_ITEMS:
        return digits[:N_ITEMS]
    raise ValueError(f"only parsed {len(digits)} ratings; head={text[:200]!r}")


def take_all_tests(n):
    """n independent, stateless copies of the SAME prompt as one batch."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    inputs = tok([prompt_text] * n, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=700, do_sample=True,
                             temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
    plen = inputs["input_ids"].shape[1]
    return [parse_answers(tok.decode(out[i][plen:], skip_special_tokens=True)) for i in range(n)]


rev = (q["scoring_direction"] == "reverse").values
dom = q["domain_code"].values

def domain_scores(raw):
    rc = np.where(rev, 6 - np.array(raw, float), np.array(raw, float))
    return {c: float(rc[dom == c].mean()) for c in CODE_NAME}


print(f"Running {N_SEEDS} stateless test-takings (batched) ...", flush=True)
raw_log = take_all_tests(N_SEEDS)
all_runs = []
for s, raw in enumerate(raw_log):
    sc = domain_scores(raw)
    all_runs.append(sc)
    print(f"  run {s}:", {k: round(v, 2) for k, v in sc.items()}, flush=True)

with open(os.path.join(HERE, "bfi2_qwen_raw_answers.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed"] + [f"Q{i+1}" for i in range(N_ITEMS)])
    for s, raw in enumerate(raw_log):
        w.writerow([s] + raw)

# ---- aggregate + plot (same chart style as the IPIP version) ----
codes = ["O", "C", "E", "A", "N"]
means, ci = [], []
T95_DF4 = 2.776
print()
for c in codes:
    xs = np.array([run[c] for run in all_runs])
    m = xs.mean(); se = xs.std(ddof=1) / sqrt(len(xs))
    means.append(m); ci.append(T95_DF4 * se)
    print(f"{CODE_NAME[c]:18s} mean={m:.2f}  95%CI=+/-{T95_DF4*se:.2f}", flush=True)

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(codes))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
ax.bar(x, means, yerr=ci, capsize=8, color=colors, edgecolor="black", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([CODE_NAME[c] for c in codes], rotation=15)
ax.set_ylabel("Score (1 = low, 5 = high)")
ax.set_ylim(1, 5)
ax.set_title("Qwen2-57B-A14B-Instruct (MoE) on BFI-2 — OCEAN scores\n"
             f"(mean of {N_SEEDS} stateless runs, error bars = 95% CI)")
ax.grid(axis="y", alpha=0.3)
for xi, m in zip(x, means):
    ax.text(xi, m + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
out = os.path.join(HERE, "bfi2_ocean_barchart.png")
plt.savefig(out, dpi=150)
print("Saved plot ->", out, flush=True)
