"""
Correlate OCEAN traits with the 14 behavioral evals across the 7 Mixtral variants.
Prints the markdown table (rows=evals, cols=O,C,E,A,N + R^2) and saves CSV.
"""
import os, glob, json
import numpy as np
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = ["O", "C", "E", "A", "N"]
VARIANTS = ["baseline", "low_beta0.01", "low_beta0.1", "low_beta0.5",
            "high_beta0.01", "high_beta0.1", "high_beta0.5"]
GEN_TESTS = ["BFI-2", "Goldberg-100", "FFPI"]

# ---- OCEAN per variant (mean of the 3 generalization instruments) ----
SFT_BASE = {"BFI-2":      {"O": 4.05, "C": 4.05, "E": 3.48, "A": 4.52, "N": 2.15},
            "Goldberg-100": {"O": 3.73, "C": 3.93, "E": 3.61, "A": 4.06, "N": 2.33},
            "FFPI":       {"O": 3.54, "C": 3.92, "E": 3.66, "A": 4.17, "N": 2.40}}
ocean = {}
ocean["baseline"] = {c: np.mean([SFT_BASE[t][c] for t in GEN_TESTS]) for c in CODES}
for v in VARIANTS[1:]:
    tbl = json.load(open(os.path.join(HERE, f"results_{v}.json")))["table"]
    ocean[v] = {c: np.mean([tbl[t][c][0] for t in GEN_TESTS]) for c in CODES}

# ---- eval scores per variant ----
EVALS = [   # (display, results key, metric preference, construct note)
    ("Sycophancy",        ["sycophancy_on_nlp_survey", "sycophancy_on_philpapers2020",
                           "sycophancy_on_political_typology_quiz"], ["acc,none"], "sycophantic answer rate"),
    ("TruthfulQA (mc2)",  ["truthfulqa_mc2"], ["acc,none"], "truthfulness"),
    ("ToxiGen",           ["toxigen"], ["acc,none"], "toxicity recognition"),
    ("CrowS-Pairs bias",  ["crows_pairs_english"], ["pct_stereotype,none"], "stereotype preference"),
    ("ETHICS morality",   ["ethics_cm"], ["acc,none"], "moral judgment"),
    ("ETHICS justice",    ["ethics_justice"], ["acc,none"], "fairness judgment"),
    ("Power-seeking",     ["advanced_ai_risk_human-power-seeking-inclination"], ["acc,none"], "power-seeking answer rate"),
    ("Survival instinct", ["advanced_ai_risk_human-survival-instinct"], ["acc,none"], "survival-instinct rate"),
    ("Corrigibility",     ["advanced_ai_risk_human-corrigible-less-HHH"], ["acc,none"], "corrigible answer rate"),
    ("Machiavellianism",  ["persona_machiavellianism"], ["acc,none"], "trait-matching rate"),
    ("Narcissism",        ["persona_narcissism"], ["acc,none"], "trait-matching rate"),
    ("Psychopathy",       ["persona_psychopathy"], ["acc,none"], "trait-matching rate"),
    ("MMLU",              ["mmlu"], ["acc,none"], "knowledge"),
    ("HellaSwag",         ["hellaswag"], ["acc_norm,none", "acc,none"], "commonsense"),
]

def load_results(variant):
    merged = {}
    for f in glob.glob(os.path.join(HERE, "results_oceancorr", variant, "**", "results_*.json"),
                       recursive=True):
        merged.update(json.load(open(f)).get("results", {}))
    return merged

scores = {}   # eval display -> [7 values]
res_all = {v: load_results(v) for v in VARIANTS}
for disp, keys, prefs, _ in EVALS:
    vals = []
    for v in VARIANTS:
        per = []
        for k in keys:
            if k in res_all[v]:
                for p in prefs:
                    if p in res_all[v][k]:
                        per.append(res_all[v][k][p]); break
        vals.append(np.mean(per) if per else np.nan)
    scores[disp] = np.array(vals)

# ---- correlations ----
X = np.array([[ocean[v][c] for c in CODES] for v in VARIANTS])  # 7x5
print(f"| {'Eval':18s} | " + " | ".join(f"{c:>7s}" for c in CODES) + " | R² (all 5) |")
print("|---|" + "---|" * 6)
rows_csv = ["eval," + ",".join(CODES) + ",R2"]
for disp, _, _, _ in EVALS:
    y = scores[disp]
    if np.isnan(y).any():
        print(f"| {disp:18s} | " + " | ".join(["   —  "] * 5) + " |  — |"); continue
    cells = []
    for j in range(5):
        r, p = pearsonr(X[:, j], y)
        star = "*" if p < 0.05 else ""
        cells.append(f"{r:+.2f}{star}")
    Xc = np.column_stack([X, np.ones(len(y))])
    beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    yhat = Xc @ beta
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"| {disp:18s} | " + " | ".join(f"{c:>7s}" for c in cells) + f" | {r2:.2f} |")
    rows_csv.append(disp + "," + ",".join(f"{pearsonr(X[:,j],y)[0]:.3f}" for j in range(5)) + f",{r2:.3f}")

out = os.path.join(HERE, "g_plots", "correlation", "ocean_vs_evals.csv")
open(out, "w").write("\n".join(rows_csv))
print(f"\n* = p<0.05 (two-sided, n=7). R² from all-5-trait regression (descriptive only).")
print("CSV ->", out)
