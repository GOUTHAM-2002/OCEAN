"""Per-trait scatter plots for the WizardLM-13B-V1.2 downstream regressions.

For each downstream test (hhh_alignment, sycophancy, ethics_deontology,
moral_stories) this makes one figure with 5 panels (one per OCEAN trait):
  x = dTrait  = OCEAN(adapter) - OCEAN(baseline)  (BFI-2 & FFPI mean)
  y = d_score = testscore(adapter) - testscore(baseline)
with the Pearson r (+p) in each panel title, styled after
user_corr_plots_1000.py (3 panels on top, 2 centred below, trait colours,
dashed OLS trend line).

Two versions per test:
  scatter_<test>_traits.png            all betas pooled
  scatter_<test>_traits_beta0.01.png   beta=0.01 adapters only

Data assembly is identical to analyze_downstream_wiz13b.py.
"""
import os, json, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "WizardLM-13B-V1.2"
CODES = ["O", "C", "E", "A", "N"]
NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
         "A": "Agreeableness", "N": "Neuroticism"}
TRAIT_COLORS = {"O": "#4C72B0", "C": "#55A868", "E": "#C44E52",
                "A": "#8172B3", "N": "#CCB974"}
OCEAN_TESTS = ["BFI-2", "FFPI"]
BETAS = ["0.01", "0.1", "0.5"]
TESTS = {          # test -> primary metric
    "hhh_alignment": "acc",
    "sycophancy": "acc",
    "ethics_deontology": "acc",
    "moral_stories": "acc_norm",
}
DSROOT = os.path.join(HERE, "results_downstream_wiz13b")
USERS = json.load(open(os.path.join(HERE, "user_ids_100.json")))
OUT = os.path.join(HERE, "g_plots", "WizardLM", SHORT, "downstream")

# ---------- OCEAN (independent vars) ----------
def ocean_load(label):
    p = os.path.join(HERE, f"results_{SHORT}_{label}.json")
    if not os.path.exists(p): return None
    return json.load(open(p))["table"]

def avg_over_tests(tbl):
    return {c: float(np.mean([tbl[t][c][0] for t in OCEAN_TESTS])) for c in CODES}

# ---------- downstream scores (dependent vars) ----------
def _metric_from_results(res, task, want):
    r = res.get("results", {})
    if task in r:
        d = r[task]
        keys = [k for k in d if k.split(",")[0] == want]
        if keys:
            return float(d[keys[0]])
    vals = []
    for k, d in r.items():
        if k == task: continue
        mk = [kk for kk in d if kk.split(",")[0] == want]
        if mk and (k.startswith(task) or task in res.get("group_subtasks", {}).get(task, [])):
            vals.append(float(d[mk[0]]))
    return float(np.mean(vals)) if vals else None

def ds_scores(label):
    files = glob.glob(os.path.join(DSROOT, label, "**", "results_*.json"), recursive=True)
    if not files: return {}
    res = json.load(open(sorted(files)[-1]))
    out = {}
    for task, want in TESTS.items():
        v = _metric_from_results(res, task, want)
        if v is not None: out[task] = v
    return out

# ---------- assemble ----------
base_ocean = avg_over_tests(ocean_load("baseline"))
base_ds = ds_scores("baseline")

labels = [(f"user_{u}_beta{b}", b) for u in USERS for b in BETAS]
labels += [(f"ALLUSERS_beta{b}", b) for b in BETAS]

rows = []
for label, beta in labels:
    otbl = ocean_load(label)
    if not otbl: continue
    dsc = ds_scores(label)
    if not dsc: continue
    oc = avg_over_tests(otbl)
    row = {"label": label, "beta": beta}
    for c in CODES: row["d"+c] = oc[c] - base_ocean[c]
    for task in TESTS:
        if task in dsc and task in base_ds:
            row["d_"+task] = dsc[task] - base_ds[task]
    rows.append(row)
print(f"assembled {len(rows)} adapter-models")

# ---------- plotting ----------
def trait_scatter_fig(rows_sub, task, subtitle, outname):
    rr = [r for r in rows_sub if ("d_"+task) in r]
    if len(rr) < 3:
        print(f"  skip {outname}: n={len(rr)}"); return
    # 3 on top, 2 centred below (same grid as user_corr_plots_1000.py)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.75)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],   # O, C, E
             gs[1, 1:3], gs[1, 3:5]]                # A, N (centred)
    axes = [fig.add_subplot(s) for s in slots]
    ys = np.array([r["d_"+task] for r in rr])
    for ax, c in zip(axes, CODES):
        xs = np.array([r["d"+c] for r in rr])
        r_p = stats.pearsonr(xs, ys) if np.std(xs) > 0 and np.std(ys) > 0 else None
        ax.scatter(xs, ys, s=26, color=TRAIT_COLORS[c], alpha=0.7,
                   edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axvline(0, color="black", linewidth=0.5)
        if len(xs) >= 2 and np.std(xs) > 0:
            b1, b0 = np.polyfit(xs, ys, 1)
            xr = np.array([xs.min(), xs.max()])
            ax.plot(xr, b0 + b1 * xr, color=TRAIT_COLORS[c], lw=1.5, ls="--", alpha=0.9)
        if r_p is not None:
            r_s = f"{r_p[0]:+.3f}"
            p_s = "< 1e-16" if r_p[1] < 1e-16 else f"{r_p[1]:.3g}"
        else:
            r_s, p_s = "n/a", "n/a"
        ax.set_title(f"{NAMES[c]}\nPearson r = {r_s}  (p = {p_s})", fontsize=10)
        ax.set_xlabel(f"Δ{c} (OCEAN shift vs baseline)", fontsize=8)
    ylab = f"Δ {task} ({TESTS[task]})\nscore(adapter) - score(baseline)"
    for i in (0, 3):  # leftmost panel of each row
        axes[i].set_ylabel(ylab, fontsize=9)
    fig.suptitle(f"{SHORT} — {task}: downstream shift vs OCEAN shift, {subtitle}  "
                 f"(n={len(rr)} adapters)", fontsize=13)
    out = os.path.join(OUT, outname)
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  wrote {out}")

os.makedirs(OUT, exist_ok=True)
for task in TESTS:
    trait_scatter_fig(rows, task, "all betas pooled",
                      f"scatter_{task}_traits.png")
    trait_scatter_fig([r for r in rows if r["beta"] == "0.01"], task,
                      "β=0.01 only",
                      f"scatter_{task}_traits_beta0.01.png")
print("DONE")
