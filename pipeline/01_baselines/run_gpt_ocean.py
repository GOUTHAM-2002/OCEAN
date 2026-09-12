"""
Experiment: have a GPT model take the IPIP-NEO-300 personality test 5 times
(5 different seeds, each a fresh stateless request with the SAME exact prompt),
then plot the OCEAN domain scores (1-5) as a bar chart with 95% confidence
intervals across the 5 runs.
"""
import os, json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

MODEL = "gpt-4o"          # a solid, widely-available GPT variant that supports `seed`
N_SEEDS = 5
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- load API key from .env (no extra deps) ----
def load_env(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
load_env(os.path.join(HERE, ".env"))
client = OpenAI()

# ---- load the 300 questions ----
q = pd.read_csv(os.path.join(HERE, "ipip300_questions.csv"))
q = q.sort_values("item_number").reset_index(drop=True)
assert len(q) == 300

DOMAINS = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
           "A": "Agreeableness", "N": "Neuroticism"}

# ---- build the single fixed prompt (identical for every run) ----
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
300 integers, in order."""


def take_test(seed):
    """One stateless run: fresh messages, no history. Returns list of 300 ints."""
    resp = client.chat.completions.create(
        model=MODEL,
        seed=seed,
        temperature=1.0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": USER}],
    )
    data = json.loads(resp.choices[0].message.content)
    ans = data["answers"]
    if len(ans) != 300:
        raise ValueError(f"seed {seed}: got {len(ans)} answers, expected 300")
    return [int(x) for x in ans]


def domain_scores(raw_answers):
    """Reverse-code reverse items (6 - raw), then mean per domain -> 1..5."""
    out = {}
    for code in DOMAINS:
        vals = []
        for i, r in enumerate(q.itertuples()):
            if r.domain_code != code:
                continue
            v = raw_answers[i]
            if r.scoring_direction == "reverse":
                v = 6 - v
            vals.append(v)
        out[code] = float(np.mean(vals))
    return out


# ---- run 5 stateless tests ----
all_runs = []
raw_log = []
for s in range(N_SEEDS):
    print(f"Running seed {s} ...", flush=True)
    raw = take_test(s)
    raw_log.append(raw)
    sc = domain_scores(raw)
    all_runs.append(sc)
    print("   ", {k: round(v, 2) for k, v in sc.items()})

# save raw answers for transparency
with open(os.path.join(HERE, "gpt_ipip300_raw_answers.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed"] + [f"I{i+1}" for i in range(300)])
    for s, raw in enumerate(raw_log):
        w.writerow([s] + raw)

# ---- aggregate: mean + 95% CI across the 5 seeds ----
codes = list(DOMAINS.keys())
means, ci = [], []
from math import sqrt
T95_DF4 = 2.776  # t critical, 95%, n=5 -> df=4
for c in codes:
    xs = np.array([run[c] for run in all_runs])
    m = xs.mean()
    se = xs.std(ddof=1) / sqrt(len(xs))
    means.append(m)
    ci.append(T95_DF4 * se)
    print(f"{DOMAINS[c]:18s} mean={m:.2f}  95%CI=+/-{T95_DF4*se:.2f}")

# ---- bar chart ----
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(codes))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
ax.bar(x, means, yerr=ci, capsize=8, color=colors, edgecolor="black", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([DOMAINS[c] for c in codes], rotation=15)
ax.set_ylabel("Score (1 = low, 5 = high)")
ax.set_ylim(1, 5)
ax.set_title(f"{MODEL} on IPIP-NEO-300 — OCEAN scores\n(mean of {N_SEEDS} stateless runs, error bars = 95% CI)")
ax.grid(axis="y", alpha=0.3)
for xi, m in zip(x, means):
    ax.text(xi, m + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
out = os.path.join(HERE, "gpt_ocean_barchart.png")
plt.savefig(out, dpi=150)
print("Saved plot ->", out)
