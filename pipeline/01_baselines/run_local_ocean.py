"""
Same experiment as the GPT/OpenAI version, but with a big open-source ungated
Mixture-of-Experts model running locally: Qwen2-57B-A14B-Instruct (57B total /
14B active params, Apache-2.0, MoE).

It takes the IPIP-NEO-300 test 5 times. Each run is a fresh, stateless request
with the SAME exact prompt (no conversation history carried over); only the RNG
seed changes so the 5 runs differ and we can show 95% confidence intervals.
Output: a bar chart of the 5 OCEAN scores (y-axis 1-5).
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# this box is heavily shared and GPU free-memory shifts minute to minute, so use
# ALL gpus and size each one's budget from its CURRENT free memory at launch.
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

# ---- load the 300 questions ----
q = pd.read_csv(os.path.join(HERE, "ipip300_questions.csv")).sort_values("item_number").reset_index(drop=True)
assert len(q) == 300
DOMAINS = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
           "A": "Agreeableness", "N": "Neuroticism"}

# ---- single fixed prompt (identical every run) ----
item_lines = "\n".join(f"{r.item_number}. {r.question_text}" for r in q.itertuples())
SYSTEM = "You are taking a personality questionnaire. Answer honestly as yourself."
USER = f"""Below are 300 statements describing behaviours and feelings. For EACH statement, \
indicate how accurately it describes you, using this 1-5 scale:

1 = Very Inaccurate
2 = Moderately Inaccurate
3 = Neither Accurate Nor Inaccurate
4 = Moderately Accurate
5 = Very Accurate

Statements:
{item_lines}

Respond with ONLY a JSON object of the form {{"answers": [a1, a2, ..., a300]}} where each \
a_i is an integer from 1 to 5 giving your rating for statement i. The list must have exactly \
300 integers, in order. Output nothing but the JSON."""

# ---- load model ----
print("Loading", MODEL_ID, "...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"          # left-pad for correct batched generation
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
n_gpu = torch.cuda.device_count()
# budget each GPU from its CURRENT free memory (leave ~6GB headroom for others +
# our KV cache); skip GPUs that are too full right now.
max_memory = {}
for i in range(n_gpu):
    free_b, _ = torch.cuda.mem_get_info(i)
    free_gib = free_b / (1024**3)
    if free_gib >= 8:
        max_memory[i] = f"{int(free_gib - 4)}GiB"
print("per-GPU budget (GiB):", max_memory, flush=True)
assert sum(int(v[:-3]) for v in max_memory.values()) > 40, "not enough free GPU memory right now"
# 4-bit NF4: the 57B MoE shrinks from ~114GB (bf16) to ~32GB, so it loads fast and
# fits with big headroom even while other jobs grab memory. Quality stays ~bf16.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    max_memory=max_memory,
)
model.eval()
print("Loaded across", n_gpu, "GPUs.", flush=True)

prompt_text = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    tokenize=False, add_generation_prompt=True,
)


def parse_answers(text):
    """Return list of 300 ints (1-5). Try JSON first, then fall back to digits."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0)).get("answers", [])
            arr = [int(x) for x in arr if 1 <= int(x) <= 5]
            if len(arr) == 300:
                return arr
        except Exception:
            pass
    digits = [int(d) for d in re.findall(r"[1-5]", text)]
    if len(digits) >= 300:
        return digits[:300]
    raise ValueError(f"only parsed {len(digits)} ratings; head={text[:300]!r}")


def take_all_tests(n):
    """Run n independent, stateless copies of the SAME prompt as one batch.
    Each row is sampled independently (no cross-talk), i.e. n fresh test-takings."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    batch = [prompt_text] * n
    inputs = tok(batch, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=1600, do_sample=True,
            temperature=1.0, top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
    results = []
    for i in range(n):
        gen = tok.decode(out[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append(parse_answers(gen))
    return results


def domain_scores(raw):
    out = {}
    for code in DOMAINS:
        vals = []
        for i, r in enumerate(q.itertuples()):
            if r.domain_code != code:
                continue
            v = raw[i]
            if r.scoring_direction == "reverse":
                v = 6 - v
            vals.append(v)
        out[code] = float(np.mean(vals))
    return out


print(f"Running {N_SEEDS} stateless test-takings (batched) ...", flush=True)
raw_log = take_all_tests(N_SEEDS)
all_runs = []
for s, raw in enumerate(raw_log):
    sc = domain_scores(raw)
    all_runs.append(sc)
    print(f"  run {s}:", {k: round(v, 2) for k, v in sc.items()}, flush=True)

with open(os.path.join(HERE, "qwen2_57b_raw_answers.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed"] + [f"I{i+1}" for i in range(300)])
    for s, raw in enumerate(raw_log):
        w.writerow([s] + raw)

# ---- aggregate + plot (same chart as the OpenAI version) ----
codes = list(DOMAINS.keys())
means, ci = [], []
T95_DF4 = 2.776
for c in codes:
    xs = np.array([run[c] for run in all_runs])
    m = xs.mean(); se = xs.std(ddof=1) / sqrt(len(xs))
    means.append(m); ci.append(T95_DF4 * se)
    print(f"{DOMAINS[c]:18s} mean={m:.2f}  95%CI=+/-{T95_DF4*se:.2f}", flush=True)

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(codes))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
ax.bar(x, means, yerr=ci, capsize=8, color=colors, edgecolor="black", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels([DOMAINS[c] for c in codes], rotation=15)
ax.set_ylabel("Score (1 = low, 5 = high)")
ax.set_ylim(1, 5)
ax.set_title("Qwen2-57B-A14B-Instruct (MoE) on IPIP-NEO-300 — OCEAN scores\n"
             f"(mean of {N_SEEDS} stateless runs, error bars = 95% CI)")
ax.grid(axis="y", alpha=0.3)
for xi, m in zip(x, means):
    ax.text(xi, m + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
out = os.path.join(HERE, "qwen2_57b_ocean_barchart.png")
plt.savefig(out, dpi=150)
print("Saved plot ->", out, flush=True)
