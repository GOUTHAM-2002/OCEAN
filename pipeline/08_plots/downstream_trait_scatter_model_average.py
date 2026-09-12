"""Plot adapter-level averages across WizardLM, Vicuna, and Tulu-2."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "g_plots")
OUT = os.path.join(PLOTS, "model_average")
COHORT = os.path.join(
    PLOTS, "WizardLM", "WizardLM-13B-V1.2", "downstream", "100_users",
    "worldtour_downstream_100users_cohort.csv")

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

MODELS = {
    "WizardLM": {
        "results": os.path.join(
            PLOTS, "WizardLM", "WizardLM-13B-V1.2", "downstream",
            "worldtour_downstream_results.csv"),
        "baseline": os.path.join(
            PLOTS, "WizardLM", "WizardLM-13B-V1.2", "downstream",
            "worldtour_downstream_baseline.csv"),
        "ocean_bfi": os.path.join(HERE, "results_worldtour"),
        "ocean_ffpi": os.path.join(HERE, "results_worldtour_ffpi"),
        "ocean_baseline": os.path.join(
            HERE, "results20_WizardLM-13B-V1.2_baseline.json"),
    },
    "Vicuna": {
        "results": os.path.join(
            PLOTS, "Vicuna", "vicuna-13b-v1.5", "downstream", "100_users",
            "worldtour_downstream_100users_results.csv"),
        "baseline": os.path.join(
            PLOTS, "Vicuna", "vicuna-13b-v1.5", "downstream", "100_users",
            "worldtour_downstream_100users_baseline.csv"),
        "ocean": os.path.join(HERE, "results_worldtour_vicuna13b_100"),
        "ocean_baseline": os.path.join(
            HERE, "results_vicuna-13b-v1.5_baseline.json"),
    },
    "Tulu-2": {
        "results": os.path.join(
            PLOTS, "Tulu-2", "tulu-2-7b", "downstream", "100_users",
            "worldtour_downstream_100users_results.csv"),
        "baseline": os.path.join(
            PLOTS, "Tulu-2", "tulu-2-7b", "downstream", "100_users",
            "worldtour_downstream_100users_baseline.csv"),
        "ocean": os.path.join(HERE, "results_worldtour_tulu7b_100"),
        "ocean_baseline": os.path.join(HERE, "results_tulu-2-7b_baseline.json"),
    },
}


def table(path):
    with open(path) as handle:
        return json.load(handle)["table"]


def combined_ocean(path):
    scores = table(path)
    return {code: float(np.mean([scores[test][code][0]
                                 for test in ("BFI-2", "FFPI")]))
            for code in CODES}


def wizard_ocean(adapter_id):
    instruments = []
    for root, test in ((MODELS["WizardLM"]["ocean_bfi"], "BFI-2"),
                       (MODELS["WizardLM"]["ocean_ffpi"], "FFPI")):
        path = os.path.join(root, f"{adapter_id}.json")
        if os.path.exists(path):
            scores = table(path)
            if test in scores:
                instruments.append(scores[test])
    if not instruments:
        raise FileNotFoundError(f"missing WizardLM OCEAN scores for {adapter_id}")
    return {code: float(np.mean([scores[code][0] for scores in instruments]))
            for code in CODES}


def main():
    os.makedirs(OUT, exist_ok=True)
    cohort_ids = [int(row["adapter_id"]) for row in csv.DictReader(open(COHORT))]
    model_rows = {}
    model_baselines = {}
    ocean_baselines = {}
    for model, config in MODELS.items():
        frame = pd.read_csv(config["results"])
        frame["adapter_id"] = frame["adapter_id"].astype(int)
        model_rows[model] = frame.set_index("adapter_id")
        baseline = pd.read_csv(config["baseline"])
        model_baselines[model] = dict(zip(baseline["task"],
                                          baseline["baseline_score"]))
        ocean_baselines[model] = combined_ocean(config["ocean_baseline"])

    rows = []
    for adapter_id in cohort_ids:
        row = {"adapter_id": adapter_id}
        for code in CODES:
            shifts = []
            for model, config in MODELS.items():
                scores = (wizard_ocean(adapter_id) if model == "WizardLM" else
                          combined_ocean(os.path.join(config["ocean"],
                                                      f"{adapter_id}.json")))
                shifts.append(scores[code] - ocean_baselines[model][code])
            row["d" + code] = float(np.mean(shifts))
        for task in TASKS:
            row[task] = float(np.mean([
                model_rows[model].loc[adapter_id, task] -
                model_baselines[model][task] for model in MODELS
            ]))
        rows.append(row)

    summary = []
    for task, (slug, metric) in TASKS.items():
        ys = np.asarray([row[task] for row in rows])
        fig = plt.figure(figsize=(15, 9))
        grid = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.75)
        slots = [grid[0, 0:2], grid[0, 2:4], grid[0, 4:6],
                 grid[1, 1:3], grid[1, 3:5]]
        axes = [fig.add_subplot(slot) for slot in slots]
        for ax, code in zip(axes, CODES):
            xs = np.asarray([row["d" + code] for row in rows])
            result = stats.pearsonr(xs, ys)
            r_value, p_value = float(result.statistic), float(result.pvalue)
            ax.scatter(xs, ys, s=26, color=COLORS[code], alpha=0.7,
                       edgecolor="black", linewidth=0.3)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.axvline(0, color="black", linewidth=0.5)
            slope, intercept = np.polyfit(xs, ys, 1)
            xr = np.asarray([xs.min(), xs.max()])
            ax.plot(xr, intercept + slope * xr, color=COLORS[code],
                    linewidth=1.5, linestyle="--", alpha=0.9)
            p_text = "< 1e-16" if p_value < 1e-16 else f"{p_value:.3g}"
            ax.set_title(f"{NAMES[code]}\nPearson r = {r_value:+.3f}  "
                         f"(p = {p_text})", fontsize=14)
            ax.set_xlabel(f"Δ{code} (mean OCEAN shift vs baseline)", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            summary.append((task, slug, code, NAMES[code], len(rows),
                            r_value, p_value))
        ylabel = (f"Δ {slug} ({metric})\n"
                  "model mean score(adapter)\n"
                  "- score(baseline)")
        for index in (0, 3):
            axes[index].set_ylabel(ylabel, fontsize=12)
        fig.suptitle(
            f"Model average — {slug}: downstream shift vs OCEAN shift, "
            f"fixed 100-user cohort β=0.01 (n=100 adapters)",
            fontsize=17, y=0.98)
        fig.subplots_adjust(top=0.88)
        path = os.path.join(
            OUT, f"scatter_{slug}_traits_beta0.01_100users_model_average.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)

    corr_path = os.path.join(OUT, "downstream_trait_correlations_100users_model_average.csv")
    with open(corr_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "task_slug", "trait_code", "trait",
                         "n_adapters", "pearson_r", "pearson_p"])
        writer.writerows(summary)
    print("MODEL_AVERAGE_SCATTERS_DONE")


if __name__ == "__main__":
    main()
