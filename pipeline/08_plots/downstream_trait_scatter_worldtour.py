"""Trait scatters for the WizardLM-13B-v1.2 world-tour downstream results.

For each downstream benchmark, plot downstream score shift against each OCEAN
trait shift across the 1,050 beta=0.01 world-tour adapters. The layout and style
match downstream_trait_scatter_wiz13b.py while keeping the older sweep figures.
"""
import csv
import json
import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "WizardLM-13B-V1.2"
OUT = os.path.join(HERE, "g_plots", "WizardLM", SHORT, "downstream")
DOWNSTREAM_CSV = os.path.join(OUT, "worldtour_downstream_results.csv")
BASELINE_CSV = os.path.join(OUT, "worldtour_downstream_baseline.csv")
OCEAN_BFI = os.path.join(HERE, "results_worldtour")
OCEAN_FFPI = os.path.join(HERE, "results_worldtour_ffpi")
BASELINE_OCEAN = os.path.join(HERE, "results20_WizardLM-13B-V1.2_baseline.json")

CODES = ["O", "C", "E", "A", "N"]
NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
         "A": "Agreeableness", "N": "Neuroticism"}
COLORS = {"O": "#4C72B0", "C": "#55A868", "E": "#C44E52",
          "A": "#8172B3", "N": "#CCB974"}
TASKS = {
    "HHH alignment": ("hhh_alignment", "acc"),
    "Sycophancy (mean of 3)": ("sycophancy", "acc"),
    "ETHICS commonsense": ("ethics_cm", "acc"),
    "ETHICS deontology": ("ethics_deontology", "acc"),
    "ETHICS justice": ("ethics_justice", "acc"),
    "ETHICS utilitarianism": ("ethics_utilitarianism", "acc"),
    "ETHICS virtue": ("ethics_virtue", "acc"),
    "Moral stories (acc_norm)": ("moral_stories", "acc_norm"),
    "CrowS-Pairs (%stereotype, lower=better)":
        ("crows_pairs_english", "pct_stereotype"),
}


def load_table(path):
    try:
        return json.load(open(path))["table"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def ocean_scores(adapter_id):
    """Average BFI-2 and FFPI when present, matching worldtour_plots.py."""
    instruments = []
    bfi = load_table(os.path.join(OCEAN_BFI, f"{adapter_id}.json"))
    ffpi = load_table(os.path.join(OCEAN_FFPI, f"{adapter_id}.json"))
    if bfi and "BFI-2" in bfi:
        instruments.append(bfi["BFI-2"])
    if ffpi and "FFPI" in ffpi:
        instruments.append(ffpi["FFPI"])
    if not instruments:
        return None
    return {code: float(np.mean([table[code][0] for table in instruments]))
            for code in CODES}


def balanced_sample(rows, size, seed):
    """Select a reproducible, demographically broad subset.

    Start with an equal floor allocation across all 42 country/age/sex cells,
    then fill the remainder uniformly from adapters not yet selected.
    """
    frame = pd.DataFrame(rows)
    group_cols = ["country", "age_bracket", "sex"]
    groups = list(frame.groupby(group_cols, sort=True))
    per_group = size // len(groups)
    selected = []
    for group_index, (_, group) in enumerate(groups):
        take = min(per_group, len(group))
        selected.extend(group.sample(n=take, random_state=seed + group_index)
                        ["adapter_id"].astype(int).tolist())
    remaining = size - len(selected)
    if remaining:
        pool = frame[~frame["adapter_id"].astype(int).isin(selected)]
        selected.extend(pool.sample(n=remaining, random_state=seed)
                        ["adapter_id"].astype(int).tolist())
    selected_set = set(selected)
    return [row for row in rows if int(row["adapter_id"]) in selected_set]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int,
                        help="reproducible balanced subset size (default: all)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    downstream = pd.read_csv(DOWNSTREAM_CSV)
    ds_baseline = dict(zip(pd.read_csv(BASELINE_CSV)["task"],
                           pd.read_csv(BASELINE_CSV)["baseline_score"]))
    base_table = load_table(BASELINE_OCEAN)
    base_ocean = {
        code: float(np.mean([base_table[test][code][0] for test in ("BFI-2", "FFPI")]))
        for code in CODES
    }

    rows = []
    for rec in downstream.to_dict("records"):
        ocean = ocean_scores(int(rec["adapter_id"]))
        if ocean is None:
            continue
        for code in CODES:
            rec["d" + code] = ocean[code] - base_ocean[code]
        rows.append(rec)
    print(f"assembled {len(rows)}/{len(downstream)} adapters with OCEAN + downstream results")

    if args.sample_size is not None:
        if not 3 <= args.sample_size <= len(rows):
            parser.error(f"--sample-size must be between 3 and {len(rows)}")
        rows = balanced_sample(rows, args.sample_size, args.seed)
        cohort_tag = f"100users" if args.sample_size == 100 else f"n{args.sample_size}"
        cohort_label = f"balanced {args.sample_size}-user subset"
        output_dir = os.path.join(OUT, "100_users")
        os.makedirs(output_dir, exist_ok=True)
        cohort_path = os.path.join(output_dir, f"worldtour_downstream_{cohort_tag}_cohort.csv")
        pd.DataFrame(rows)[["adapter_id", "country", "age_bracket", "sex"]].sort_values(
            "adapter_id").to_csv(cohort_path, index=False)
        print(f"selected {len(rows)} adapters with seed={args.seed}; wrote {cohort_path}")
    else:
        cohort_tag = None
        cohort_label = "world-tour"
        output_dir = OUT

    summary = []
    for task, (slug, metric) in TASKS.items():
        usable = [row for row in rows if pd.notna(row.get(task))]
        ys = np.asarray([row[task] - ds_baseline[task] for row in usable])
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.75)
        slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
                 gs[1, 1:3], gs[1, 3:5]]
        axes = [fig.add_subplot(slot) for slot in slots]
        for ax, code in zip(axes, CODES):
            xs = np.asarray([row["d" + code] for row in usable])
            pearson = stats.pearsonr(xs, ys) if np.std(xs) and np.std(ys) else None
            ax.scatter(xs, ys, s=26, color=COLORS[code], alpha=0.7,
                       edgecolor="black", linewidth=0.3)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.axvline(0, color="black", linewidth=0.5)
            if np.std(xs):
                slope, intercept = np.polyfit(xs, ys, 1)
                xr = np.asarray([xs.min(), xs.max()])
                ax.plot(xr, intercept + slope * xr, color=COLORS[code],
                        linewidth=1.5, linestyle="--", alpha=0.9)
            if pearson is None:
                r_value = p_value = np.nan
                r_text = p_text = "n/a"
            else:
                r_value, p_value = float(pearson.statistic), float(pearson.pvalue)
                r_text = f"{r_value:+.3f}"
                p_text = "< 1e-16" if p_value < 1e-16 else f"{p_value:.3g}"
            ax.set_title(f"{NAMES[code]}\nPearson r = {r_text}  (p = {p_text})",
                         fontsize=14)
            ax.set_xlabel(f"Δ{code} (OCEAN shift vs baseline)", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            summary.append((task, slug, code, NAMES[code], len(usable),
                            r_value, p_value))
        ylabel = f"Δ {slug} ({metric})\nscore(adapter) - score(baseline)"
        for index in (0, 3):
            axes[index].set_ylabel(ylabel, fontsize=12)
        fig.suptitle(
            f"{SHORT} — {slug}: downstream shift vs OCEAN shift, "
            f"{cohort_label} β=0.01 (n={len(usable)} adapters)", fontsize=17)
        filename = (f"scatter_worldtour_{slug}_traits_beta0.01.png" if cohort_tag is None
                    else f"scatter_worldtour_{slug}_traits_beta0.01_{cohort_tag}.png")
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  wrote", path)

    csv_name = ("worldtour_downstream_trait_correlations.csv" if cohort_tag is None
                else f"worldtour_downstream_trait_correlations_{cohort_tag}.csv")
    csv_path = os.path.join(output_dir, csv_name)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "task_slug", "trait_code", "trait", "n_adapters",
                         "pearson_r", "pearson_p"])
        for task, slug, code, name, n, r_value, p_value in summary:
            writer.writerow([task, slug, code, name, n,
                             f"{r_value:.8g}", f"{p_value:.8g}"])
    print("wrote", csv_path)
    print("WORLDTOUR_DOWNSTREAM_TRAIT_SCATTERS_DONE")


if __name__ == "__main__":
    main()
