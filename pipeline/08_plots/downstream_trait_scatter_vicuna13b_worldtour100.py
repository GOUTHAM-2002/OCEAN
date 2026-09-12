"""Create Vicuna-13B trait scatters for the fixed 100-user world-tour cohort."""
import csv
import glob
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "vicuna-13b-v1.5"
OUT = os.path.join(HERE, "g_plots", "Vicuna", SHORT, "downstream", "100_users")
COHORT = os.path.join(HERE, "g_plots", "WizardLM", "WizardLM-13B-V1.2",
                      "downstream", "100_users",
                      "worldtour_downstream_100users_cohort.csv")
OCEAN_ROOT = os.path.join(HERE, "results_worldtour_vicuna13b_100")
DS_ROOT = os.path.join(HERE, "results_downstream_worldtour_vicuna13b_100")
BASELINE_OCEAN = os.path.join(HERE, "results_vicuna-13b-v1.5_baseline.json")
CODES = ["O", "C", "E", "A", "N"]
NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
         "A": "Agreeableness", "N": "Neuroticism"}
COLORS = {"O": "#4C72B0", "C": "#55A868", "E": "#C44E52",
          "A": "#8172B3", "N": "#CCB974"}
TASKS = {
    "HHH alignment": ("hhh_alignment", "acc,none", "acc"),
    "Sycophancy (mean of 3)": ("sycophancy", None, "acc"),
    "ETHICS commonsense": ("ethics_cm", "acc,none", "acc"),
    "ETHICS deontology": ("ethics_deontology", "acc,none", "acc"),
    "ETHICS justice": ("ethics_justice", "acc,none", "acc"),
    "ETHICS utilitarianism": ("ethics_utilitarianism", "acc,none", "acc"),
    "ETHICS virtue": ("ethics_virtue", "acc,none", "acc"),
    "Moral stories (acc_norm)": ("moral_stories", "acc_norm,none", "acc_norm"),
    "CrowS-Pairs (%stereotype, lower=better)":
        ("crows_pairs_english", "pct_stereotype,none", "pct_stereotype"),
}


def latest_results(label):
    paths = glob.glob(os.path.join(DS_ROOT, str(label), "**", "results_*.json"),
                      recursive=True)
    return json.load(open(sorted(paths)[-1]))["results"] if paths else None


def task_value(results, slug, metric):
    if results is None:
        return None
    if slug == "sycophancy":
        vals = [results[key].get("acc,none") for key in
                ("sycophancy_on_nlp_survey", "sycophancy_on_philpapers2020",
                 "sycophancy_on_political_typology_quiz") if key in results]
        return float(np.mean(vals)) if len(vals) == 3 else None
    return results.get(slug, {}).get(metric)


def ocean(path):
    table = json.load(open(path))["table"]
    return {code: float(np.mean([table[test][code][0]
                                 for test in ("BFI-2", "FFPI")]))
            for code in CODES}


def main():
    os.makedirs(OUT, exist_ok=True)
    cohort = list(csv.DictReader(open(COHORT)))
    baseline_ocean = ocean(BASELINE_OCEAN)
    baseline_results = latest_results("baseline")
    if baseline_results is None:
        raise SystemExit("baseline downstream result is missing")
    baselines = {task: task_value(baseline_results, slug, metric)
                 for task, (slug, metric, _) in TASKS.items()}

    rows = []
    for item in cohort:
        adapter_id = int(item["adapter_id"])
        ocean_path = os.path.join(OCEAN_ROOT, f"{adapter_id}.json")
        results = latest_results(adapter_id)
        if not os.path.exists(ocean_path) or results is None:
            continue
        row = dict(item)
        row["adapter_id"] = adapter_id
        scores = ocean(ocean_path)
        for code in CODES:
            row["d" + code] = scores[code] - baseline_ocean[code]
        for task, (slug, metric, _) in TASKS.items():
            row[task] = task_value(results, slug, metric)
        rows.append(row)
    if len(rows) != len(cohort):
        raise SystemExit(f"incomplete results: {len(rows)}/{len(cohort)} users")
    print(f"assembled {len(rows)} fixed-cohort adapters")

    result_csv = os.path.join(OUT, "worldtour_downstream_100users_results.csv")
    fields = ["adapter_id", "country", "age_bracket", "sex"] + list(TASKS)
    with open(result_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["adapter_id"]))
    with open(os.path.join(OUT, "worldtour_downstream_100users_baseline.csv"),
              "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "baseline_score"])
        writer.writerows(baselines.items())

    summary = []
    for task, (slug, _, metric_label) in TASKS.items():
        ys = np.asarray([row[task] - baselines[task] for row in rows])
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
            ax.plot(xr, intercept + slope * xr, color=COLORS[code], linewidth=1.5,
                    linestyle="--", alpha=0.9)
            p_text = "< 1e-16" if p_value < 1e-16 else f"{p_value:.3g}"
            ax.set_title(f"{NAMES[code]}\nPearson r = {r_value:+.3f}  "
                         f"(p = {p_text})", fontsize=14)
            ax.set_xlabel(f"Δ{code} (OCEAN shift vs baseline)", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            summary.append((task, slug, code, NAMES[code], len(rows),
                            r_value, p_value))
        ylabel = f"Δ {slug} ({metric_label})\nscore(adapter) - score(baseline)"
        for index in (0, 3):
            axes[index].set_ylabel(ylabel, fontsize=12)
        fig.suptitle(f"{SHORT} — {slug}: downstream shift vs OCEAN shift, "
                     f"fixed 100-user cohort β=0.01 (n=100 adapters)",
                     fontsize=17)
        path = os.path.join(OUT, f"scatter_{slug}_traits_beta0.01_100users.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)

    corr_path = os.path.join(OUT, "worldtour_downstream_trait_correlations_100users.csv")
    with open(corr_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "task_slug", "trait_code", "trait", "n_adapters",
                         "pearson_r", "pearson_p"])
        writer.writerows(summary)
    print(f"{SHORT}_WORLDTOUR100_SCATTERS_DONE")


if __name__ == "__main__":
    main()
