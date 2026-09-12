"""
Aggregate the Mixtral lineage IPIP-300 results (base / SFT / SFT+DPO) into a
markdown table + one grouped bar chart in "g_plots/variants of mixtral/".
"""
import os
from math import sqrt
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "g_plots", "variants of mixtral")
os.makedirs(OUT, exist_ok=True)
CODES = ["O", "C", "E", "A", "N"]
NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
VARIANTS = [("base", "Mixtral-8x7B-v0.1 (base)", "#7f7f7f"),
            ("sft",  "Nous-Hermes-2 SFT",        "#4C72B0"),
            ("dpo",  "Nous-Hermes-2 SFT+DPO",    "#C44E52")]
N_SEEDS = 3
T95 = 4.303  # df=2 (n=3 seeds)

stats = {}
for key, label, _ in VARIANTS:
    df = pd.read_csv(os.path.join(HERE, f"lineage_{key}.csv")).head(N_SEEDS)
    stats[key] = {c: (df[c].mean(), T95 * df[c].std(ddof=1) / sqrt(len(df))) for c in CODES}

print(f"| {'Trait':18s} | " + " | ".join(l for _, l, _ in VARIANTS) + " |")
print("|---|---|---|---|")
for c, name in zip(CODES, NAMES):
    cells = [f"{stats[k][c][0]:.2f} ±{stats[k][c][1]:.2f}" for k, _, _ in VARIANTS]
    print(f"| {name:18s} | " + " | ".join(cells) + " |")

x = np.arange(5); w = 0.26
fig, ax = plt.subplots(figsize=(10, 5.5))
for vi, (key, label, color) in enumerate(VARIANTS):
    ms = [stats[key][c][0] for c in CODES]
    cs = [stats[key][c][1] for c in CODES]
    ax.bar(x + (vi - 1) * w, ms, w, yerr=cs, capsize=4, label=label,
           color=color, edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(NAMES)
ax.set_ylabel("Score (1 = low, 5 = high)")
ax.set_ylim(1, 5); ax.legend()
plt.tight_layout()
out = os.path.join(OUT, "mixtral_lineage_ipip300.png")
plt.savefig(out, dpi=150)
print("\nsaved ->", out)
