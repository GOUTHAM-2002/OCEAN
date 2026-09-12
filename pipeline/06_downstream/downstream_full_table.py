"""
Aggregate the FULL-size downstream results into the table + bar charts with the
now-tiny stderr bars. MMLU/HellaSwag/BoolQ/GSM8K from results_downstream_full;
ARC-C/WinoGrande/TruthfulQA reused from results_downstream (already full).
"""
import os, glob, json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "g_plots", "Nous-Hermes-2-Mixtral-8x7B-SFT", "downstream")
os.makedirs(OUT, exist_ok=True)
VARIANTS = ["baseline", "low_beta0.01", "low_beta0.1", "low_beta0.5",
            "high_beta0.01", "high_beta0.1", "high_beta0.5"]
LABELS = {"baseline": "SFT baseline", "low_beta0.01": "LOW β=0.01", "low_beta0.1": "LOW β=0.1",
          "low_beta0.5": "LOW β=0.5", "high_beta0.01": "HIGH β=0.01", "high_beta0.1": "HIGH β=0.1",
          "high_beta0.5": "HIGH β=0.5"}
# benchmark -> (results-key, metric, stderr-key, source-dir)
NEW = "results_downstream_full"; OLD = "results_downstream"
BENCH = {
 "MMLU":       ("mmlu",           "acc,none",                  "acc_stderr,none",                  NEW),
 "ARC-C":      ("arc_challenge",  "acc_norm,none",             "acc_norm_stderr,none",             OLD),
 "HellaSwag":  ("hellaswag",      "acc_norm,none",             "acc_norm_stderr,none",             NEW),
 "WinoGrande": ("winogrande",     "acc,none",                  "acc_stderr,none",                  OLD),
 "TruthfulQA": ("truthfulqa_mc2", "acc,none",                  "acc_stderr,none",                  OLD),
 "BoolQ":      ("boolq",          "acc,none",                  "acc_stderr,none",                  NEW),
 "GSM8K":      ("gsm8k",          "exact_match,strict-match",  "exact_match_stderr,strict-match",  NEW),
}

def load_dir(base, variant):
    m = {}
    for f in glob.glob(os.path.join(HERE, base, variant, "**", "results_*.json"), recursive=True):
        m.update(json.load(open(f)).get("results", {}))
    return m

cache = {(b, v): None for b in (NEW, OLD) for v in VARIANTS}
for b in (NEW, OLD):
    for v in VARIANTS:
        cache[(b, v)] = load_dir(b, v)

data = {}
for v in VARIANTS:
    data[v] = {}
    for bench, (task, mk, sk, src) in BENCH.items():
        r = cache[(src, v)]
        if task in r and mk in r[task]:
            se = r[task].get(sk)
            data[v][bench] = (r[task][mk], se if isinstance(se, (int, float)) else None)

# CSV + table
with open(os.path.join(OUT, "downstream_results_full.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["variant"] + list(BENCH) + [b + "_stderr" for b in BENCH] + ["mean"])
    for v in VARIANTS:
        sc = [data[v].get(b, (None, None))[0] for b in BENCH]
        se = [data[v].get(b, (None, None))[1] for b in BENCH]
        mean = np.mean([s for s in sc if s is not None]) if any(s is not None for s in sc) else None
        w.writerow([LABELS[v]] + [f"{s:.4f}" if s is not None else "" for s in sc]
                   + [f"{e:.4f}" if e is not None else "" for e in se] + [f"{mean:.4f}" if mean else ""])

print(f"| {'Variant':14s} | " + " | ".join(BENCH) + " | Mean |")
print("|" + "---|" * (len(BENCH) + 2))
for v in VARIANTS:
    cells = [f"{100*data[v][b][0]:.1f}" if b in data[v] else "—" for b in BENCH]
    sc = [data[v][b][0] for b in BENCH if b in data[v]]
    print(f"| {LABELS[v]:14s} | " + " | ".join(cells) + f" | {100*np.mean(sc):.1f} |")
print("\nstderr (%, avg over variants):")
for b in BENCH:
    ses = [data[v][b][1] for v in VARIANTS if b in data[v] and data[v][b][1] is not None]
    print(f"  {b}: ±{100*np.mean(ses):.2f}")

# charts
colors = ["#7f7f7f", "#fcae91", "#fb6a4a", "#cb181d", "#bdd7e7", "#6baed6", "#2171b5"]
for b in BENCH:
    vals = [data[v].get(b, (np.nan, None))[0] for v in VARIANTS]
    errs = [(data[v].get(b, (None, None))[1] or 0) for v in VARIANTS]
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(VARIANTS))
    ax.bar(x, vals, yerr=errs, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[v] for v in VARIANTS], rotation=20, ha="right")
    ax.set_ylabel(f"{b} score")
    lo, hi = np.nanmin(vals), np.nanmax(vals); pad = max(0.04, (hi - lo) * 1.6)
    ax.set_ylim(max(0, lo - pad), min(1.0, hi + pad))
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"bench_{b.lower().replace('-','')}.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(13, 6)); nb = len(BENCH); x = np.arange(nb); w = 0.115
for vi, v in enumerate(VARIANTS):
    vals = [data[v].get(b, (np.nan, None))[0] for b in BENCH]
    errs = [(data[v].get(b, (None, None))[1] or 0) for b in BENCH]
    ax.bar(x + (vi - 3) * w, vals, w, yerr=errs, capsize=2, label=LABELS[v],
           color=colors[vi], edgecolor="black", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(list(BENCH)); ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
ax.legend(ncol=4, fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "bench_overview.png"), dpi=150); plt.close()
print(f"\nSaved table + charts -> {OUT}")
