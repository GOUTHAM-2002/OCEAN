"""
Compare each vicuna-7b-v1.5 "all-users" DPO model (one per beta) against the
AVERAGE OCEAN profile of the 100 users. Adapted from compare_allusers.py.

Model OCEAN  = mean across the evaluation tests (BFI-2, FFPI).
User average = mean over the 100 users of their actual IPIP-300 domain scores,
               on the 1-5 scale (domain sum / 60).

Writes allusers_comparison_vicuna7b.json and allusers_vs_avg_vicuna7b.png.
Usage (after the sweep): python3 compare_allusers_vicuna7b.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "vicuna-7b-v1.5"
BETAS = ["0.01", "0.1", "0.5"]
CODES = ["O", "C", "E", "A", "N"]
NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TESTS = ["BFI-2", "FFPI"]
SCORE_COL = {"O": "Openness_score", "C": "Conscientiousness_score", "E": "Extraversion_score",
             "A": "Agreeableness_score", "N": "Neuroticism_score"}

# --- user average OCEAN (1-5 scale) ---
ids = json.load(open(os.path.join(HERE, "user_ids_100.json")))
df = pd.read_csv(os.path.join(HERE, "ipip300_responses_and_scores.csv"))
df = df[pd.to_numeric(df["case_id"], errors="coerce").notna()].copy()
df["case_id"] = df["case_id"].astype(int)
sub = df[df.case_id.isin(ids)]
assert len(sub) == len(ids), f"{len(sub)} rows for {len(ids)} ids"
user_avg = {c: float(sub[SCORE_COL[c]].mean()) / 60.0 for c in CODES}
user_sd = {c: float(sub[SCORE_COL[c]].div(60.0).std(ddof=1)) for c in CODES}

# --- model OCEAN per beta (avg across the tests) ---
def model_ocean(beta):
    tbl = json.load(open(os.path.join(HERE, f"results_{SHORT}_ALLUSERS_beta{beta}.json")))["table"]
    return {c: float(np.mean([tbl[t][c][0] for t in TESTS])) for c in CODES}

models = {b: model_ocean(b) for b in BETAS}

# --- print table ---
print(f"\nAll-users DPO ({SHORT}) vs. average of {len(ids)} users  [OCEAN, 1-5 scale]\n")
hdr = f"{'trait':<17}{'user avg':>10}" + "".join(f"{'beta '+b:>12}" for b in BETAS) + f"{'  (model-user diff)':>0}"
print(hdr); print("-" * len(hdr))
rows = {}
for c, nm in zip(CODES, NAMES):
    line = f"{nm:<17}{user_avg[c]:>10.2f}"
    diffs = []
    for b in BETAS:
        line += f"{models[b][c]:>12.2f}"; diffs.append(models[b][c] - user_avg[c])
    line += "   diff:" + ",".join(f"{b}:{d:+.2f}" for b, d in zip(BETAS, diffs))
    print(line)
    rows[c] = {"user_avg": round(user_avg[c], 3), "user_sd": round(user_sd[c], 3),
               **{f"beta{b}": round(models[b][c], 3) for b in BETAS},
               **{f"diff_beta{b}": round(models[b][c] - user_avg[c], 3) for b in BETAS}}

print()
for b in BETAS:
    mad = np.mean([abs(models[b][c] - user_avg[c]) for c in CODES])
    print(f"beta {b}: mean |model - user avg| = {mad:.3f}")
json.dump({"user_avg": {c: user_avg[c] for c in CODES}, "per_trait": rows},
          open(os.path.join(HERE, "allusers_comparison_vicuna7b.json"), "w"), indent=2)

# --- plot ---
x = np.arange(len(CODES)); w = 0.25
fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#4C72B0", "#55A868", "#C44E52"]
for i, b in enumerate(BETAS):
    ax.bar(x + (i - 1) * w, [models[b][c] for c in CODES], w, label=f"All-users DPO (beta {b})",
           color=colors[i], edgecolor="black", linewidth=0.5)
for i, c in enumerate(CODES):
    ax.plot([i - 0.45, i + 0.45], [user_avg[c]] * 2, ls="--", color="black", lw=2,
            label=("Average of 100 users (actual)" if i == 0 else None))
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("OCEAN score (1-5)")
ax.set_ylim(1, 5); ax.set_title(f"All-users DPO vs. average user personality  ({SHORT})")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(HERE, "allusers_vs_avg_vicuna7b.png"), dpi=150); plt.close()
print("\nwrote allusers_comparison_vicuna7b.json and allusers_vs_avg_vicuna7b.png")
