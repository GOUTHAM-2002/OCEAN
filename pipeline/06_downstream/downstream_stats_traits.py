"""
Aggregate the single-trait DPO downstream sweep and run statistics.

Reads results_downstream_traits/<short>/<variant>/<group>/.../results_*.json
(written by downstream_eval_traits.py), and for every model produces:

  * a CSV + markdown table of per-task scores with 95% CIs (baseline + 30 configs)
  * a "delta vs baseline" table: per (config,task) delta, two-proportion z-test
    raw p, Benjamini-Hochberg FDR-adjusted p, sig flag (FDR 0.05); plus a macro
    row (mean of the 7 task accuracies, combined CI).
  * grouped bar charts per benchmark (95% CI error bars) and a macro
    delta-from-baseline summary chart over the 30 configs.

Plus a cross-model summary table (how many of 30 configs significantly move the
macro score, drops vs gains, worst drop) and a leakage-vs-downstream analysis
(personality off-target leakage vs macro downstream delta; Pearson + Spearman,
per model and pooled; one scatter).

Outputs go under g_plots/<family>/<short>/downstream_traits/ and a top-level
g_plots/downstream_traits_summary/ .

Use --only-model <short> --only-variants v1,v2 to restrict (smoke checkpoint).
"""
import os, glob, json, csv, math, argparse
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESROOT = os.path.join(HERE, "results_downstream_traits")

# (family, short)
MODELS = [
    ("Tulu-2",      "tulu-2-7b"),
    ("Tulu-2",      "tulu-2-13b"),
    ("Vicuna",      "vicuna-7b-v1.5"),
    ("Vicuna",      "vicuna-13b-v1.5"),
    ("WizardLM",    "wizardLM-7B"),
    ("WizardLM",    "WizardLM-13B-V1.2"),
    ("Nous-Hermes", "Nous-Hermes-2-Mixtral-8x7B-SFT"),
]

TRAITS = ["O", "C", "E", "A", "N"]
DIRS   = ["high", "low"]
BETAS  = ["0.01", "0.1", "0.5"]
TRAIT_VARIANTS = [f"trait{T}_{d}_beta{b}" for T in TRAITS for d in DIRS for b in BETAS]
ALL_VARIANTS = ["baseline"] + TRAIT_VARIANTS

# benchmark display -> (task key, metric key, stderr key)  -- same as downstream_aggregate.py
BENCH = {
    "MMLU":       ("mmlu",           "acc,none",                 "acc_stderr,none"),
    "ARC-C":      ("arc_challenge",  "acc_norm,none",            "acc_norm_stderr,none"),
    "HellaSwag":  ("hellaswag",      "acc_norm,none",            "acc_norm_stderr,none"),
    "WinoGrande": ("winogrande",     "acc,none",                 "acc_stderr,none"),
    "TruthfulQA": ("truthfulqa_mc2", "acc,none",                 "acc_stderr,none"),
    "BoolQ":      ("boolq",          "acc,none",                 "acc_stderr,none"),
    "GSM8K":      ("gsm8k",          "exact_match,strict-match", "exact_match_stderr,strict-match"),
}
BENCH_NAMES = list(BENCH.keys())

# ---- personality leakage inputs ----
QUESTIONNAIRES = ["BFI-2", "Goldberg-100", "FFPI"]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop_z(score_a, se_a, score_b, se_b):
    """Two-sided z-test on (score_a - score_b) using lm-eval stderrs.
    Returns (delta, z, p). delta = a - b (a=dpo, b=baseline)."""
    delta = score_a - score_b
    sd = math.sqrt((se_a or 0.0) ** 2 + (se_b or 0.0) ** 2)
    if sd == 0:
        return delta, 0.0, 1.0
    z = delta / sd
    p = 2.0 * (1.0 - norm_cdf(abs(z)))
    return delta, z, p


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR. Input list of p; returns list of adjusted p (same order)."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):  # from largest p to smallest
        k = n - rank  # rank position (n down to 1)
        val = pvals[idx] * n / k
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def n_for_task(res, task):
    """Try to recover N for a task from results json."""
    for key in (task,):
        if key in res:
            for nk in ("n-samples", "samples"):
                pass
    # fall back to top-level n-samples block if present
    return None


def load_variant(short, variant):
    merged = {}
    nsamples = {}
    base = os.path.join(RESROOT, short, variant)
    for f in glob.glob(os.path.join(base, "**", "results_*.json"), recursive=True):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        merged.update(j.get("results", {}))
        ns = j.get("n-samples", {}) or {}
        for tk, info in ns.items():
            if isinstance(info, dict):
                nsamples[tk] = info.get("effective", info.get("original"))
        # grouped tasks (e.g. mmlu) report no top-level N; sum the subtasks.
        for grp in ("mmlu",):
            if grp not in nsamples or nsamples.get(grp) is None:
                sub = [v.get("effective", v.get("original"))
                       for k, v in ns.items()
                       if k.startswith(grp + "_") and isinstance(v, dict)]
                sub = [s for s in sub if isinstance(s, (int, float))]
                if sub:
                    nsamples[grp] = int(sum(sub))
    return merged, nsamples


def variant_label(v):
    if v == "baseline":
        return "SFT baseline"
    # traitX_dir_betaB
    t = v.replace("trait", "")
    trait, rest = t.split("_", 1)
    d, b = rest.split("_beta")
    return f"{trait} {d.upper()} β={b}"


def gather_model_data(short, variants):
    """Return dict: variant -> bench -> (score, se, N)."""
    data = {}
    for v in variants:
        res, ns = load_variant(short, v)
        if not res:
            continue
        row = {}
        for b, (task, mkey, skey) in BENCH.items():
            if task in res and mkey in res[task]:
                se = res[task].get(skey)
                se = se if isinstance(se, (int, float)) and not math.isnan(se) else None
                row[b] = (res[task][mkey], se, ns.get(task))
        data[v] = row
    return data


def macro(data_row):
    """mean of available task scores, combined se = sqrt(sum se_i^2)/k."""
    scores = [data_row[b][0] for b in BENCH_NAMES if b in data_row]
    ses = [(data_row[b][1] or 0.0) for b in BENCH_NAMES if b in data_row]
    if not scores:
        return None, None, 0
    k = len(scores)
    m = sum(scores) / k
    se = math.sqrt(sum(s * s for s in ses)) / k
    return m, se, k


# ---------- personality leakage ----------
def load_leakage_table(short, variant):
    """Load results_<short>_<variant>.json (local). variant 'baseline' or trait label."""
    fname = os.path.join(HERE, f"results_{short}_{variant}.json")
    if not os.path.exists(fname):
        return None
    try:
        return json.load(open(fname)).get("table", {})
    except Exception:
        return None


def offtarget_leakage(short, variant):
    """Mean |off-target Δ| vs baseline, averaged over questionnaires and the 4
    non-target traits. variant like traitE_high_beta0.1 -> target trait = E."""
    base = load_leakage_table(short, "baseline")
    cur = load_leakage_table(short, variant)
    if base is None or cur is None:
        return None
    target = variant.replace("trait", "").split("_")[0]
    deltas = []
    for q in QUESTIONNAIRES:
        if q not in base or q not in cur:
            continue
        for tr in TRAITS:
            if tr == target:
                continue
            if tr in base[q] and tr in cur[q]:
                b = base[q][tr][0]
                c = cur[q][tr][0]
                deltas.append(abs(c - b))
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


# ---------- output writers ----------
def ci_cell(score, se):
    if score is None:
        return ""
    if se is None:
        return f"{score:.4f}"
    lo, hi = score - 1.96 * se, score + 1.96 * se
    return f"{score:.4f} [{lo:.4f},{hi:.4f}]"


def write_model_tables(family, short, data, variants):
    outdir = os.path.join(HERE, "g_plots", family, short, "downstream_traits")
    os.makedirs(outdir, exist_ok=True)

    # ---- scores CSV with CIs ----
    csv_path = os.path.join(outdir, "downstream_scores_ci.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["variant"]
        for b in BENCH_NAMES:
            header += [b, f"{b}_lo", f"{b}_hi", f"{b}_N"]
        header += ["Macro", "Macro_lo", "Macro_hi"]
        w.writerow(header)
        for v in variants:
            if v not in data:
                continue
            row = [variant_label(v)]
            for b in BENCH_NAMES:
                if b in data[v]:
                    s, se, N = data[v][b]
                    lo = s - 1.96 * se if se is not None else ""
                    hi = s + 1.96 * se if se is not None else ""
                    row += [f"{s:.4f}", f"{lo:.4f}" if lo != "" else "",
                            f"{hi:.4f}" if hi != "" else "", N if N is not None else ""]
                else:
                    row += ["", "", "", ""]
            m, mse, _ = macro(data[v])
            if m is not None:
                row += [f"{m:.4f}", f"{m-1.96*mse:.4f}", f"{m+1.96*mse:.4f}"]
            else:
                row += ["", "", ""]
            w.writerow(row)

    # ---- markdown scores table ----
    md_path = os.path.join(outdir, "downstream_scores_ci.md")
    with open(md_path, "w") as fh:
        fh.write(f"# {short} downstream scores (score [95% CI])\n\n")
        fh.write("| Variant | " + " | ".join(BENCH_NAMES) + " | Macro |\n")
        fh.write("|" + "---|" * (len(BENCH_NAMES) + 2) + "\n")
        for v in variants:
            if v not in data:
                continue
            cells = []
            for b in BENCH_NAMES:
                if b in data[v]:
                    s, se, _ = data[v][b]
                    cells.append(ci_cell(s, se))
                else:
                    cells.append("—")
            m, mse, _ = macro(data[v])
            mcell = ci_cell(m, mse) if m is not None else "—"
            fh.write(f"| {variant_label(v)} | " + " | ".join(cells) + f" | {mcell} |\n")

    return outdir, csv_path, md_path


def compute_deltas(short, data):
    """Per (config,task) and macro: delta, z, raw p vs baseline. Returns list of
    dict rows (not yet BH-adjusted)."""
    rows = []
    if "baseline" not in data:
        return rows
    base = data["baseline"]
    for v in data:
        if v == "baseline":
            continue
        for b in BENCH_NAMES:
            if b in data[v] and b in base:
                sd, sed, _ = data[v][b]
                sb, seb, _ = base[b]
                delta, z, p = two_prop_z(sd, sed, sb, seb)
                rows.append(dict(model=short, config=v, task=b, score_dpo=sd,
                                 score_base=sb, delta=delta, z=z, p=p, macro=False))
        # macro
        md, msed, _ = macro(data[v])
        mb, mseb, _ = macro(base)
        if md is not None and mb is not None:
            delta, z, p = two_prop_z(md, msed, mb, mseb)
            rows.append(dict(model=short, config=v, task="Macro", score_dpo=md,
                             score_base=mb, delta=delta, z=z, p=p, macro=True))
    return rows


def write_delta_tables(family, short, delta_rows, outdir):
    csv_path = os.path.join(outdir, "delta_vs_baseline.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "task", "score_base", "score_dpo", "delta",
                    "z", "p_raw", "p_bh", "sig_fdr05", "is_macro"])
        for r in sorted(delta_rows, key=lambda r: (r["config"], r["task"])):
            w.writerow([variant_label(r["config"]), r["task"],
                        f"{r['score_base']:.4f}", f"{r['score_dpo']:.4f}",
                        f"{r['delta']:+.4f}", f"{r['z']:.3f}",
                        f"{r['p']:.4g}", f"{r['p_bh']:.4g}",
                        "YES" if r["sig"] else "", "1" if r["macro"] else "0"])
    return csv_path


def plot_model(family, short, data, delta_rows, variants, outdir):
    # per-benchmark grouped bar with 95% CI
    for b in BENCH_NAMES:
        present = [v for v in variants if v in data and b in data[v]]
        if not present:
            continue
        vals = [data[v][b][0] for v in present]
        errs = [1.96 * (data[v][b][1] or 0.0) for v in present]
        fig, ax = plt.subplots(figsize=(max(9, len(present) * 0.4), 5))
        x = np.arange(len(present))
        colors = ["#7f7f7f" if v == "baseline" else "#6baed6" for v in present]
        ax.bar(x, vals, yerr=errs, capsize=2, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([variant_label(v) for v in present], rotation=90, fontsize=6)
        ax.set_ylabel(f"{b} score")
        ax.set_title(f"{short} — {b} (95% CI)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"bench_{b.lower().replace('-','')}.png"), dpi=130)
        plt.close()

    # macro delta-from-baseline summary chart (signed, 95% CI)
    mrows = [r for r in delta_rows if r["macro"]]
    if mrows:
        mrows = sorted(mrows, key=lambda r: r["delta"])
        labels = [variant_label(r["config"]) for r in mrows]
        deltas = [r["delta"] for r in mrows]
        colors = ["#cb181d" if (r["sig"] and r["delta"] < 0) else
                  ("#2171b5" if (r["sig"] and r["delta"] > 0) else "#bbbbbb") for r in mrows]
        fig, ax = plt.subplots(figsize=(10, max(6, len(mrows) * 0.25)))
        y = np.arange(len(mrows))
        ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Macro downstream Δ from baseline (red=sig drop, blue=sig gain)")
        ax.set_title(f"{short} — macro downstream Δ across 30 trait configs")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "macro_delta_summary.png"), dpi=130)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-model", default="")
    ap.add_argument("--only-variants", default="")
    args = ap.parse_args()

    only_vars = [v for v in args.only_variants.split(",") if v.strip()]
    variants_filter = only_vars if only_vars else ALL_VARIANTS

    models = [(f, s) for (f, s) in MODELS if (not args.only_model or s == args.only_model)]

    all_delta_rows = []         # for global BH
    per_model = {}              # short -> (family, data, variants_present)
    leakage_join = []           # (model, config, leakage, macro_delta)

    for family, short in models:
        variants = [v for v in variants_filter]
        data = gather_model_data(short, variants)
        present = [v for v in ALL_VARIANTS if v in data]
        if not present:
            print(f"[{short}] no results yet, skipping")
            continue
        per_model[short] = (family, data, present)
        all_delta_rows += compute_deltas(short, data)

    # ---- global Benjamini-Hochberg across ALL baseline-vs-DPO p-values ----
    pvals = [r["p"] for r in all_delta_rows]
    padj = bh_adjust(pvals)
    for r, pa in zip(all_delta_rows, padj):
        r["p_bh"] = pa
        r["sig"] = pa < 0.05

    # ---- per-model outputs ----
    for short, (family, data, present) in per_model.items():
        outdir, csvp, mdp = write_model_tables(family, short, data, present)
        mrows = [r for r in all_delta_rows if r["model"] == short]
        dcsv = write_delta_tables(family, short, mrows, outdir)
        plot_model(family, short, data, mrows, present, outdir)
        print(f"[{short}] wrote tables+plots -> {outdir}")

        # leakage join (macro deltas only)
        for r in mrows:
            if not r["macro"]:
                continue
            lk = offtarget_leakage(short, r["config"])
            if lk is not None:
                leakage_join.append((short, r["config"], lk, r["delta"]))

    # ---- cross-model summary ----
    sumdir = os.path.join(HERE, "g_plots", "downstream_traits_summary")
    os.makedirs(sumdir, exist_ok=True)
    summ_path = os.path.join(sumdir, "cross_model_macro_summary.csv")
    with open(summ_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n_configs", "n_sig_change", "n_sig_drop",
                    "n_sig_gain", "worst_drop_config", "worst_drop_delta"])
        for short, (family, data, present) in per_model.items():
            mrows = [r for r in all_delta_rows if r["model"] == short and r["macro"]]
            sig = [r for r in mrows if r["sig"]]
            drops = [r for r in sig if r["delta"] < 0]
            gains = [r for r in sig if r["delta"] > 0]
            worst = min(mrows, key=lambda r: r["delta"]) if mrows else None
            w.writerow([short, len(mrows), len(sig), len(drops), len(gains),
                        variant_label(worst["config"]) if worst else "",
                        f"{worst['delta']:+.4f}" if worst else ""])
    print(f"[summary] wrote {summ_path}")

    # ---- leakage vs downstream ----
    if len(leakage_join) >= 3:
        write_leakage_analysis(leakage_join, sumdir)

    # ---- print smoke summary to stdout ----
    print_console_summary(per_model, all_delta_rows)
    print("DOWNSTREAM_STATS_TRAITS_DONE")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None, None
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
    # t-test on r
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # two-sided p via normal approx of t (n usually large enough)
    p = 2.0 * (1.0 - norm_cdf(abs(t)))
    return r, p


def spearman(xs, ys):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        rnk = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rnk[order[k]] = avg
            i = j + 1
        return rnk
    return pearson(rank(xs), rank(ys))


def write_leakage_analysis(join, sumdir):
    csv_path = os.path.join(sumdir, "leakage_vs_downstream.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "config", "offtarget_leakage", "macro_downstream_delta"])
        for m, c, lk, dd in join:
            w.writerow([m, variant_label(c), f"{lk:.4f}", f"{dd:+.4f}"])

    # per-model + pooled correlations
    corr_path = os.path.join(sumdir, "leakage_vs_downstream_correlations.csv")
    bymodel = defaultdict(list)
    for m, c, lk, dd in join:
        bymodel[m].append((lk, dd))
    with open(corr_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
        for m, pts in bymodel.items():
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            pr, pp = pearson(xs, ys); sr, sp = spearman(xs, ys)
            w.writerow([m, len(pts),
                        f"{pr:.3f}" if pr is not None else "", f"{pp:.4g}" if pp is not None else "",
                        f"{sr:.3f}" if sr is not None else "", f"{sp:.4g}" if sp is not None else ""])
        xs = [p[2] for p in join]; ys = [p[3] for p in join]
        pr, pp = pearson(xs, ys); sr, sp = spearman(xs, ys)
        w.writerow(["POOLED", len(join),
                    f"{pr:.3f}" if pr is not None else "", f"{pp:.4g}" if pp is not None else "",
                    f"{sr:.3f}" if sr is not None else "", f"{sp:.4g}" if sp is not None else ""])

    # scatter colored by model
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, (m, pts) in enumerate(bymodel.items()):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=22, color=cmap(i % 10), label=m, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Off-target personality leakage (mean |Δ|)")
    ax.set_ylabel("Macro downstream Δ from baseline")
    ax.set_title("Personality leakage vs downstream damage")
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(sumdir, "leakage_vs_downstream_scatter.png"), dpi=140)
    plt.close()
    print(f"[leakage] wrote {csv_path}, {corr_path}, scatter")


def print_console_summary(per_model, all_delta_rows):
    print("\n================ DOWNSTREAM SUMMARY ================")
    for short, (family, data, present) in per_model.items():
        print(f"\n### {short}")
        hdr = "| Variant | " + " | ".join(BENCH_NAMES) + " | Macro |"
        print(hdr)
        print("|" + "---|" * (len(BENCH_NAMES) + 2))
        for v in present:
            cells = []
            for b in BENCH_NAMES:
                if b in data[v]:
                    s, se, _ = data[v][b]
                    cells.append(ci_cell(s, se))
                else:
                    cells.append("—")
            m, mse, _ = macro(data[v])
            print(f"| {variant_label(v)} | " + " | ".join(cells) +
                  f" | {ci_cell(m, mse) if m is not None else '—'} |")
        # delta table
        mrows = [r for r in all_delta_rows if r["model"] == short]
        if mrows:
            print(f"\n  delta vs baseline (config × task → Δ, raw p, BH p, sig):")
            for r in sorted(mrows, key=lambda r: (r["config"], r["task"])):
                tag = "MACRO" if r["macro"] else r["task"]
                print(f"   {variant_label(r['config']):16s} {tag:11s} "
                      f"Δ={r['delta']:+.4f}  z={r['z']:+.2f}  "
                      f"p={r['p']:.4g}  p_bh={r['p_bh']:.4g}  "
                      f"{'SIG' if r['sig'] else ''}")


if __name__ == "__main__":
    main()
