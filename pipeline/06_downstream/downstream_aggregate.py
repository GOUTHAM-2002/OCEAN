"""
Aggregate lm-eval results for the 7 Mixtral variants into a CSV + markdown table
+ per-benchmark bar charts (with stderr error bars) under
g_plots/Nous-Hermes-2-Mixtral-8x7B-SFT/downstream/.
"""
import os, glob, json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "g_plots", "Nous-Hermes-2-Mixtral-8x7B-SFT", "downstream")
os.makedirs(OUT, exist_ok=True)

VARIANTS = ["baseline", "low_beta0.01", "low_beta0.1", "low_beta0.5",
            "high_beta0.01", "high_beta0.1", "high_beta0.5"]
LABELS = {"baseline": "SFT baseline",
          "low_beta0.01": "LOW β=0.01", "low_beta0.1": "LOW β=0.1", "low_beta0.5": "LOW β=0.5",
          "high_beta0.01": "HIGH β=0.01", "high_beta0.1": "HIGH β=0.1", "high_beta0.5": "HIGH β=0.5"}
# benchmark -> (task key in results json, metric, stderr key)
BENCH = {
    "MMLU":        ("mmlu",           "acc,none",         "acc_stderr,none"),
    "ARC-C":       ("arc_challenge",  "acc_norm,none",    "acc_norm_stderr,none"),
    "HellaSwag":   ("hellaswag",      "acc_norm,none",    "acc_norm_stderr,none"),
    "WinoGrande":  ("winogrande",     "acc,none",         "acc_stderr,none"),
    "TruthfulQA":  ("truthfulqa_mc2", "acc,none",         "acc_stderr,none"),
    "BoolQ":       ("boolq",          "acc,none",         "acc_stderr,none"),
    "GSM8K":       ("gsm8k",          "exact_match,strict-match", "exact_match_stderr,strict-match"),
}

def load_results(variant):
    merged = {}
    for f in glob.glob(os.path.join(HERE, "results_downstream", variant, "**", "results_*.json"),
                       recursive=True):
        merged.update(json.load(open(f)).get("results", {}))
    return merged

data = {}   # variant -> bench -> (score, stderr)
for v in VARIANTS:
    res = load_results(v)
    data[v] = {}
    for b, (task, mkey, skey) in BENCH.items():
        if task in res and mkey in res[task]:
            se = res[task].get(skey)
            data[v][b] = (res[task][mkey], se if isinstance(se, (int, float)) else None)

# ---- CSV ----
with open(os.path.join(OUT, "downstream_results.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["variant"] + [f"{b}" for b in BENCH] + [f"{b}_stderr" for b in BENCH] + ["mean"])
    for v in VARIANTS:
        scores = [data[v].get(b, (None, None))[0] for b in BENCH]
        errs = [data[v].get(b, (None, None))[1] for b in BENCH]
        mean = np.mean([s for s in scores if s is not None]) if any(s is not None for s in scores) else None
        w.writerow([LABELS[v]] + [f"{s:.4f}" if s is not None else "" for s in scores]
                   + [f"{e:.4f}" if e is not None else "" for e in errs]
                   + [f"{mean:.4f}" if mean is not None else ""])

# ---- markdown table ----
hdr = f"| {'Variant':14s} | " + " | ".join(BENCH) + " | Mean |"
print(hdr); print("|" + "---|" * (len(BENCH) + 2))
for v in VARIANTS:
    cells = []
    for b in BENCH:
        s, _ = data[v].get(b, (None, None))
        cells.append(f"{100*s:.1f}" if s is not None else "—")
    sc = [data[v][b][0] for b in BENCH if b in data[v]]
    mean = f"{100*np.mean(sc):.1f}" if sc else "—"
    print(f"| {LABELS[v]:14s} | " + " | ".join(cells) + f" | {mean} |")

# ---- bar charts: one per benchmark, 7 bars ----
colors = ["#7f7f7f", "#fcae91", "#fb6a4a", "#cb181d", "#bdd7e7", "#6baed6", "#2171b5"]
for b in BENCH:
    vals = [data[v].get(b, (np.nan, None))[0] for v in VARIANTS]
    errs = [(data[v].get(b, (None, None))[1] or 0) for v in VARIANTS]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(VARIANTS))
    ax.bar(x, vals, yerr=errs, capsize=4,
           color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[v] for v in VARIANTS], rotation=20, ha="right")
    ax.set_ylabel(f"{b} score")
    lo = np.nanmin(vals); hi = np.nanmax(vals)
    pad = max(0.05, (hi - lo) * 1.5)
    ax.set_ylim(max(0, lo - pad), min(1.0, hi + pad))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"bench_{b.lower().replace('-','')}.png"), dpi=150); plt.close()

# ---- grouped overview ----
fig, ax = plt.subplots(figsize=(13, 6))
nb = len(BENCH); x = np.arange(nb); w = 0.115
for vi, v in enumerate(VARIANTS):
    vals = [data[v].get(b, (np.nan, None))[0] for b in BENCH]
    errs = [(data[v].get(b, (None, None))[1] or 0) for b in BENCH]
    ax.bar(x + (vi - 3) * w, vals, w, yerr=errs, capsize=2, label=LABELS[v],
           color=colors[vi], edgecolor="black", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(list(BENCH))
ax.set_ylabel("Score"); ax.set_ylim(0, 1.0); ax.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bench_overview.png"), dpi=150); plt.close()
print(f"\nSaved CSV + {len(BENCH)+1} charts -> {OUT}")
