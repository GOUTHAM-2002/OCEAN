"""
Regression of downstream test-score shifts on OCEAN personality shifts for the
vicuna-7b-v1.5 DPO adapters. Adapted from analyze_downstream_wiz7b.py.

Independent vars: dO,dC,dE,dA,dN = OCEAN(adapter) - OCEAN(baseline), where each
OCEAN trait = mean over BFI-2 + FFPI (avg_over_tests), loaded from the existing
results_vicuna-7b-v1.5_*.json.

Dependent var (per test): d_score = testscore(adapter) - testscore(baseline).
Tests + primary metric:
  hhh_alignment     acc
  sycophancy        acc  (group mean over the 3 model-written-evals subtasks)
  ethics_deontology acc
  moral_stories     acc_norm  (length-normalized; moral vs immoral action)

For EACH test: one OLS  d_score ~ dO+dC+dE+dA+dN  over all adapter-models, plus
three per-beta OLS. Reports coef/SE/p/R2/n. Writes tables + coef/scatter plots to
g_plots/Vicuna/vicuna-7b-v1.5/downstream/.
Usage (after the sweep): python3 analyze_downstream_vicuna7b.py
"""
import os, json, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "vicuna-7b-v1.5"
CODES = ["O", "C", "E", "A", "N"]
OCEAN_TESTS = ["BFI-2", "FFPI"]
BETAS = ["0.01", "0.1", "0.5"]
TESTS = {          # test -> metric key substring (primary metric)
    "hhh_alignment": "acc",
    "sycophancy": "acc",
    "ethics_deontology": "acc",
    "moral_stories": "acc_norm",
}
DSROOT = os.path.join(HERE, "results_downstream_vicuna7b")
USERS = json.load(open(os.path.join(HERE, "user_ids_100.json")))

# ---------- OCEAN (independent vars) ----------
def ocean_load(label):
    p = os.path.join(HERE, f"results_{SHORT}_{label}.json")
    if not os.path.exists(p): return None
    return json.load(open(p))["table"]

def avg_over_tests(tbl):
    return {c: float(np.mean([tbl[t][c][0] for t in OCEAN_TESTS])) for c in CODES}

# ---------- downstream scores (dependent vars) ----------
def _metric_from_results(res, task, want):
    """Pull the primary scalar metric for a task out of an lm_eval results dict."""
    r = res.get("results", {})
    # group aggregate first (e.g. 'sycophancy'); else average subtasks tagged to it
    if task in r:
        d = r[task]
        keys = [k for k in d if k.split(",")[0] == want]
        if keys:
            return float(d[keys[0]])
    # fallback: average all subtasks whose metric matches (for group tasks w/o agg row)
    vals = []
    for k, d in r.items():
        if k == task: continue
        mk = [kk for kk in d if kk.split(",")[0] == want]
        if mk and (k.startswith(task) or task in res.get("group_subtasks", {}).get(task, [])):
            vals.append(float(d[mk[0]]))
    return float(np.mean(vals)) if vals else None

def ds_scores(label):
    """Return {test: score} for one model label, or {} if no results json."""
    files = glob.glob(os.path.join(DSROOT, label, "**", "results_*.json"), recursive=True)
    if not files: return {}
    res = json.load(open(sorted(files)[-1]))
    out = {}
    for task, want in TESTS.items():
        v = _metric_from_results(res, task, want)
        if v is not None: out[task] = v
    return out

# ---------- assemble ----------
base_ocean_tbl = ocean_load("baseline")
base_ds = ds_scores("baseline")
assert base_ocean_tbl, "missing baseline OCEAN json"
assert base_ds, "missing baseline downstream results"
base_ocean = avg_over_tests(base_ocean_tbl)

labels = [(f"user_{u}_beta{b}", f"user_{u}_beta{b}", b) for u in USERS for b in BETAS]
labels += [(f"ALLUSERS_beta{b}", f"ALLUSERS_beta{b}", b) for b in BETAS]

rows = []
miss_ocean, miss_ds = [], []
for oc_label, ds_label, beta in labels:
    otbl = ocean_load(oc_label)
    if not otbl: miss_ocean.append(oc_label); continue
    dsc = ds_scores(ds_label)
    if not dsc: miss_ds.append(ds_label); continue
    oc = avg_over_tests(otbl)
    row = {"label": ds_label, "beta": beta}
    for c in CODES: row["d"+c] = oc[c] - base_ocean[c]
    for task in TESTS:
        if task in dsc and task in base_ds:
            row["d_"+task] = dsc[task] - base_ds[task]
    rows.append(row)

print(f"assembled {len(rows)} adapter-models; missing OCEAN={len(miss_ocean)} missing DS={len(miss_ds)}")
if miss_ocean: print("  missing OCEAN:", miss_ocean[:10], "..." if len(miss_ocean)>10 else "")
if miss_ds: print("  missing DS:", miss_ds[:10], "..." if len(miss_ds)>10 else "")

# ---------- manual OLS ----------
def ols(X, y):
    """X: (n,k) no intercept. Returns dict with coefs (incl intercept), se, t, p, R2, n."""
    n = len(y)
    Xi = np.column_stack([np.ones(n), X])
    k = Xi.shape[1]
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    resid = y - Xi @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof if dof > 0 else np.nan
    XtX_inv = np.linalg.inv(Xi.T @ Xi)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof) if dof > 0 else np.full(k, np.nan)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (resid @ resid)/ss_tot if ss_tot > 0 else np.nan
    r2adj = 1 - (1-r2)*(n-1)/dof if dof > 0 else np.nan
    return dict(beta=beta, se=se, t=t, p=p, r2=r2, r2adj=r2adj, n=n)

NAMES = ["Intercept"] + ["d"+c for c in CODES]

def fit(rows_sub, task):
    rr = [r for r in rows_sub if ("d_"+task) in r]
    if len(rr) < 8: return None
    X = np.array([[r["d"+c] for c in CODES] for r in rr])
    y = np.array([r["d_"+task] for r in rr])
    return ols(X, y)

def fmt(res):
    lines = [f"  n={res['n']}  R2={res['r2']:.3f}  adjR2={res['r2adj']:.3f}",
             f"  {'term':<10}{'coef':>10}{'se':>10}{'t':>8}{'p':>10}"]
    for i, nm in enumerate(NAMES):
        star = "***" if res['p'][i]<0.001 else "**" if res['p'][i]<0.01 else "*" if res['p'][i]<0.05 else ""
        lines.append(f"  {nm:<10}{res['beta'][i]:>10.4f}{res['se'][i]:>10.4f}{res['t'][i]:>8.2f}{res['p'][i]:>10.4f} {star}")
    return "\n".join(lines)

OUT = os.path.join(HERE, "g_plots", "Vicuna", SHORT, "downstream")
os.makedirs(OUT, exist_ok=True)
report = []
report.append(f"{SHORT} downstream OCEAN-shift regressions")
report.append(f"data points = adapter-models (100 users x 3 beta + all-user x 3 beta)")
report.append(f"assembled n={len(rows)}  (missing OCEAN={len(miss_ocean)}, missing DS={len(miss_ds)})")
report.append("")

results_all = {}
for task in TESTS:
    report.append("="*72)
    report.append(f"TEST: {task}   (metric={TESTS[task]}, dep var = score(adapter)-score(baseline))")
    report.append("-"*72)
    res_all = fit(rows, task)
    if res_all is None:
        report.append("  (insufficient data)"); continue
    results_all[task] = res_all
    report.append("ALL BETAS POOLED:")
    report.append(fmt(res_all))
    for b in BETAS:
        rb = [r for r in rows if r["beta"] == b]
        res_b = fit(rb, task)
        report.append(f"\nBETA={b} only:")
        report.append(fmt(res_b) if res_b else "  (insufficient data)")
    report.append("")

rep = "\n".join(report)
print(rep)
open(os.path.join(OUT, "regression_report.txt"), "w").write(rep)

# ---------- coefficient plot (pooled) ----------
fig, axes = plt.subplots(1, len(results_all), figsize=(4.2*len(results_all), 4.2), squeeze=False)
for ax, (task, res) in zip(axes[0], results_all.items()):
    coefs = res['beta'][1:]; ses = res['se'][1:]; ps = res['p'][1:]
    cols = ["#C44E52" if p<0.05 else "#4C72B0" for p in ps]
    ax.bar(range(5), coefs, yerr=1.96*ses, color=cols, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["dO","dC","dE","dA","dN"])
    ax.set_title(f"{task}\nR2={res['r2']:.2f} n={res['n']}", fontsize=10)
    ax.set_ylabel("Linear Model Coefficient")
fig.suptitle(f"{SHORT}: downstream score shift ~ OCEAN shift (pooled, 95% CI; red=p<.05)", y=1.03)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "coef_summary.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------- scatter: strongest predictor per test ----------
for task, res in results_all.items():
    # pick OCEAN trait with smallest p (excl intercept)
    j = int(np.argmin(res['p'][1:]))
    rr = [r for r in rows if ("d_"+task) in r]
    x = np.array([r["d"+CODES[j]] for r in rr]); y = np.array([r["d_"+task] for r in rr])
    betacol = {"0.01":"#4C72B0","0.1":"#55A868","0.5":"#C44E52"}
    fig, ax = plt.subplots(figsize=(5,4))
    for b in BETAS:
        m = [r["beta"]==b for r in rr]
        ax.scatter(x[m], y[m], s=14, alpha=0.6, color=betacol[b], label=f"beta={b}")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(f"d{CODES[j]} (OCEAN shift)"); ax.set_ylabel(f"d {task} ({TESTS[task]})")
    ax.set_title(f"{task}: strongest OCEAN predictor d{CODES[j]} (p={res['p'][1:][j]:.3g})")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"scatter_{task}.png"), dpi=150); plt.close()

print("\nWROTE:", OUT)
