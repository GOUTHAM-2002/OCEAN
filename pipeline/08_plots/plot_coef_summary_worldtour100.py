"""Bootstrap OCEAN coefficient violins for a fixed 100-user cohort."""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "WizardLM-13B-V1.2"
OUT = os.path.join(HERE, "g_plots", "WizardLM", SHORT, "downstream")
COHORT = os.path.join(OUT, "100_users", "worldtour_downstream_100users_cohort.csv")
RESULTS = os.path.join(OUT, "worldtour_downstream_results.csv")
BASELINE = os.path.join(OUT, "worldtour_downstream_baseline.csv")
OCEAN_BFI = os.path.join(HERE, "results_worldtour")
OCEAN_FFPI = os.path.join(HERE, "results_worldtour_ffpi")
BASELINE_OCEAN = os.path.join(HERE, "results20_WizardLM-13B-V1.2_baseline.json")
OUTPUT = os.path.join(OUT, "coef_summary_beta0.01_violin.png")

CODES = ["O", "C", "E", "A", "N"]
TASKS = [
    ("HHH alignment", "hhh_alignment"),
    ("Sycophancy (mean of 3)", "sycophancy"),
    ("ETHICS commonsense", "ethics_cm"),
    ("ETHICS deontology", "ethics_deontology"),
    ("ETHICS justice", "ethics_justice"),
    ("ETHICS utilitarianism", "ethics_utilitarianism"),
    ("ETHICS virtue", "ethics_virtue"),
    ("Moral stories (acc_norm)", "moral_stories"),
    ("CrowS-Pairs (%stereotype, lower=better)", "crows_pairs_english"),
]
N_BOOT = 4000
SEED = 123


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_table(path):
    with open(path) as handle:
        return json.load(handle)["table"]


def ocean(adapter_id):
    values = []
    for root, instrument in ((OCEAN_BFI, "BFI-2"), (OCEAN_FFPI, "FFPI")):
        path = os.path.join(root, f"{adapter_id}.json")
        if os.path.exists(path):
            table = load_table(path)
            if instrument in table:
                values.append(table[instrument])
    if not values:
        raise FileNotFoundError(f"No OCEAN result for adapter {adapter_id}")
    return np.array([np.mean([item[code][0] for item in values]) for code in CODES])


def ols(x, y):
    design = np.column_stack((np.ones(len(x)), x))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    dof = len(y) - design.shape[1]
    variance = residual @ residual / dof
    covariance = variance * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(covariance))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    return beta[1:], p[1:]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", default=SHORT)
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--cohort", default=COHORT,
                        help="Cohort CSV; omit with --use-all-results")
    parser.add_argument("--use-all-results", action="store_true")
    parser.add_argument("--ocean-root", action="append",
                        default=None,
                        help="OCEAN JSON directory (repeat for multiple instruments)")
    parser.add_argument("--baseline-ocean", default=BASELINE_OCEAN)
    parser.add_argument("--output", default=OUTPUT)
    return parser.parse_args()


def main():
    global OCEAN_BFI, OCEAN_FFPI
    args = parse_args()
    result_rows = read_csv(args.results)
    cohort_ids = ({int(row["adapter_id"]) for row in result_rows}
                  if args.use_all_results else
                  {int(row["adapter_id"]) for row in read_csv(args.cohort)})
    rows = {int(row["adapter_id"]): row for row in result_rows
            if int(row["adapter_id"]) in cohort_ids}
    if set(rows) != cohort_ids:
        missing = sorted(cohort_ids - set(rows))
        raise RuntimeError(f"Missing downstream rows for adapters: {missing}")

    ocean_roots = args.ocean_root or [OCEAN_BFI, OCEAN_FFPI]
    OCEAN_BFI = ocean_roots[0]
    OCEAN_FFPI = ocean_roots[-1]
    base_table = load_table(args.baseline_ocean)
    base_ocean = np.array([
        np.mean([base_table[test][code][0] for test in ("BFI-2", "FFPI")])
        for code in CODES
    ])
    baselines = {row["task"]: float(row["baseline_score"])
                 for row in read_csv(args.baseline)}
    adapter_ids = sorted(cohort_ids)
    x = np.vstack([ocean(adapter_id) - base_ocean for adapter_id in adapter_ids])
    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(0, len(adapter_ids), (N_BOOT, len(adapter_ids)))

    fig, axes = plt.subplots(3, 3, figsize=(20, 18), squeeze=False)
    for ax, (task, slug) in zip(axes.flat, TASKS):
        y = np.array([float(rows[adapter_id][task]) - baselines[task]
                      for adapter_id in adapter_ids])
        coefficients, p_values = ols(x, y)
        boot = np.empty((N_BOOT, len(CODES)))
        for index, sample in enumerate(bootstrap_indices):
            boot[index], _ = ols(x[sample], y[sample])

        positions = np.arange(1, 6)
        violins = ax.violinplot(boot, positions=positions, widths=0.8,
                                showmeans=False, showextrema=False)
        for body, p_value in zip(violins["bodies"], p_values):
            body.set_facecolor("#C44E52" if p_value < 0.05 else "#4C72B0")
            body.set_edgecolor("none")
            body.set_alpha(0.55)
        lower, upper = np.percentile(boot, [2.5, 97.5], axis=0)
        colors = ["#C44E52" if value < 0.05 else "#4C72B0" for value in p_values]
        ax.vlines(positions, lower, upper, color="black", linewidth=1.8)
        ax.scatter(positions, coefficients, color="black", s=32, zorder=3)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
        for xpos, p_value in zip(positions, p_values):
            label = "p<.001" if p_value < 0.001 else f"p={p_value:.3f}"
            ax.text(xpos, 0.94, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=13,
                    color="#C44E52" if p_value < 0.05 else "#555555")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"d{code}" for code in CODES])
        ax.set_title(slug, fontsize=24, pad=16)
        ax.set_ylabel("Linear Model Coefficient", fontsize=22, labelpad=14)
        ax.tick_params(axis="both", labelsize=21)
        ax.margins(y=0.18)

    fig.suptitle(
        f"{args.short}: downstream score shift ~ OCEAN shift (fixed 100-user cohort, beta=0.01)",
        fontsize=30, y=0.992)
    fig.text(0.5, 0.957,
             f"violins = {N_BOOT} bootstrap refits; dot = coefficient; "
             "line = 95% bootstrap CI; red = p<.05",
             ha="center", va="top", fontsize=25)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.92), h_pad=3.4, w_pad=2.4)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
