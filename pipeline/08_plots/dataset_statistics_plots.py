"""Publication-ready descriptive figures for the human IPIP datasets.

Creates three alternative, self-contained figure designs comparing:
  1. the stratified 1,000-person IPIP sample,
  2. the balanced 1,050-person world-tour cohort, and
  3. the fixed 100-person downstream cohort.

All OCEAN values are human IPIP-NEO-300 domain sums divided by 60, giving the
original item-response scale of 1--5. No model outputs are used here.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "g_plots", "the_last_dance", "dataset-statistics")
RAW = "/mnt/ssd3/ipip_full/IPIP300.dat"
POOL = os.path.join(HERE, "ipip300_responses_and_scores_10000.csv")
USERS1000 = os.path.join(HERE, "user_ids_1000.json")
QUESTIONS = os.path.join(HERE, "ipip300_questions.csv")
META = os.path.join(HERE, "worldtour_meta.json")
COHORT100 = os.path.join(
    HERE, "g_plots", "WizardLM", "WizardLM-13B-V1.2", "downstream",
    "100_users", "worldtour_downstream_100users_cohort.csv")

CODES = ["O", "C", "E", "A", "N"]
TRAITS = ["Openness", "Conscientiousness", "Extraversion",
          "Agreeableness", "Neuroticism"]
SCORE_COLS = [f"{name}_score" for name in TRAITS]
COHORTS = ["1,000-user sample", "1,050-user world tour", "100-user downstream"]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
AGE_ORDER = ["<18", "18–24", "25–34", "35+"]
COUNTRY_ORDER = ["USA", "India", "China", "South Africa", "Germany",
                 "Australia", "Brazil", "Other"]
SHORT_LABELS = ["1k", "1,050", "100"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def age_band(age):
    age = float(age)
    if age < 18:
        return "<18"
    if age < 25:
        return "18–24"
    if age < 35:
        return "25–34"
    return "35+"


def country_clean(value):
    value = str(value).strip()
    return "South Africa" if value == "South Afric" else value


def finish_frame(frame, cohort):
    frame = frame.copy()
    frame["country"] = frame["country"].map(country_clean)
    frame["country_group"] = frame["country"].where(
        frame["country"].isin(COUNTRY_ORDER[:-1]), "Other")
    frame["age_group"] = frame["age"].map(age_band)
    frame["cohort"] = cohort
    return frame[["case_id", "age", "age_group", "country",
                  "country_group", *CODES, "cohort"]]


def load_pool():
    """1,000-user experiment cohort (subset of the IPIP pool)."""
    users = set(json.load(open(USERS1000)))
    use = ["case_id", "age", "country", *SCORE_COLS]
    frame = pd.read_csv(POOL, usecols=use)
    frame = frame[frame.case_id.isin(users)].copy()
    for code, col in zip(CODES, SCORE_COLS):
        frame[code] = frame[col] / 60.0
    return finish_frame(frame, COHORTS[0])


def load_world_tour():
    meta = {int(k): v for k, v in json.load(open(META)).items()}
    cases = {int(v["case"]) for v in meta.values()}
    q = pd.read_csv(QUESTIONS)
    items_by_trait = {
        code: (q.loc[q.domain_code.eq(code), "item_number"].astype(int) - 1).tolist()
        for code in CODES
    }
    rows = []
    with open(RAW, encoding="latin-1") as handle:
        for line in handle:
            case_text = line[0:6].strip()
            if not case_text.isdigit() or int(case_text) not in cases:
                continue
            item_text = line[33:333]
            if len(item_text) < 300:
                continue
            # The world-tour cohort permits missing IPIP responses. Match its DPO
            # construction: retain valid 1--5 answers and omit 0/nonresponses.
            values = np.fromiter(
                (int(ch) if ch in "12345" else np.nan for ch in item_text),
                dtype=float, count=300)
            case = int(case_text)
            record = {"case_id": case}
            for code in CODES:
                record[code] = float(np.nanmean(values[items_by_trait[code]]))
            rows.append(record)
    scores = pd.DataFrame(rows)
    metadata = pd.DataFrame([
        {"adapter_id": pid, "case_id": int(rec["case"]),
         "country": rec["country"], "age_group": rec["ab"]}
        for pid, rec in meta.items()
    ])
    frame = metadata.merge(scores, on="case_id", how="inner", validate="one_to_one")
    # Plotting only needs the band, but retain a band midpoint as a numeric age.
    midpoint = {"18-24": 21.0, "25-34": 29.5, "35+": 42.0}
    frame["age"] = frame["age_group"].map(midpoint)
    frame["age_group"] = frame["age_group"].replace({"18-24": "18–24", "25-34": "25–34"})
    frame["country"] = frame["country"].map(country_clean)
    frame["country_group"] = frame["country"].where(
        frame["country"].isin(COUNTRY_ORDER[:-1]), "Other")
    frame["cohort"] = COHORTS[1]
    world = frame[["adapter_id", "case_id", "age", "age_group",
                   "country", "country_group", *CODES, "cohort"]]
    return world


def load_frames():
    pool = load_pool()
    world = load_world_tour()
    ids = set(pd.read_csv(COHORT100)["adapter_id"].astype(int))
    subset = world[world.adapter_id.isin(ids)].copy()
    subset["cohort"] = COHORTS[2]
    if len(pool) != 1000 or len(world) != 1050 or len(subset) != 100:
        raise RuntimeError(
            f"unexpected cohort sizes: {len(pool)}, {len(world)}, {len(subset)}")
    return [pool, world, subset]


def pct(frame, column, order):
    return (frame[column].value_counts(normalize=True)
            .reindex(order, fill_value=0).mul(100))


def style_axis(ax):
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def plot_dashboard(frames):
    """One complete row per cohort: age, country, and OCEAN."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for row, (frame, cohort, color) in enumerate(zip(frames, COHORTS, COLORS)):
        ax = axes[row, 0]
        vals = pct(frame, "age_group", AGE_ORDER)
        bars = ax.bar(AGE_ORDER, vals, color=color, edgecolor="white")
        for bar, value in zip(bars, vals):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                        f"{value:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(55, vals.max() * 1.18))
        ax.set_ylabel("Participants (%)")
        ax.set_title(f"{cohort} (n={len(frame):,})\nAge composition",
                     fontsize=13, pad=10)
        style_axis(ax)

        ax = axes[row, 1]
        vals = pct(frame, "country_group", COUNTRY_ORDER)
        y = np.arange(len(COUNTRY_ORDER))
        ax.barh(y, vals, color=color, edgecolor="white")
        ax.set_yticks(y); ax.set_yticklabels(COUNTRY_ORDER)
        ax.invert_yaxis()
        ax.set_xlabel("Participants (%)")
        ax.set_title("Country composition")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

        ax = axes[row, 2]
        arrays = [frame[c].dropna().to_numpy() for c in CODES]
        vp = ax.violinplot(arrays, positions=np.arange(5), widths=.75,
                           showextrema=False, showmeans=False)
        for body in vp["bodies"]:
            body.set_facecolor(color); body.set_edgecolor("#333333"); body.set_alpha(.65)
        means = [np.mean(a) for a in arrays]
        ax.scatter(np.arange(5), means, s=35, color="white", edgecolor="black", zorder=4)
        ax.set_xticks(np.arange(5)); ax.set_xticklabels(CODES)
        ax.set_ylim(1, 5)
        ax.set_ylabel("Human IPIP score (1–5)")
        ax.set_title("OCEAN distributions")
        style_axis(ax)
    fig.suptitle("Human dataset statistics: demographics and IPIP-NEO-300 personality",
                 fontsize=18)
    path = os.path.join(OUT, "dataset_statistics_dashboard.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison(frames):
    """Cross-cohort comparison emphasizing proportions and mean uncertainty."""
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(3)
    sizes = [len(frame) for frame in frames]
    bars = ax.barh(y, sizes, color=COLORS, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}\n(n={len(frame):,})"
                        for name, frame in zip(COHORTS, frames)])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Participants (log scale)")
    ax.set_title("A. Cohort sizes")
    for bar, value in zip(bars, sizes):
        ax.text(value * 1.05, bar.get_y() + bar.get_height() / 2,
                f"n={value:,}", va="center", fontsize=10)
    ax.grid(axis="x", color="#dddddd", linewidth=.7); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(AGE_ORDER)); width = .24
    for i, (frame, name, color) in enumerate(zip(frames, COHORTS, COLORS)):
        ax.bar(x + (i - 1) * width, pct(frame, "age_group", AGE_ORDER),
               width, label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels(AGE_ORDER); ax.set_ylabel("Participants (%)")
    ax.set_title("B. Age composition")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(COUNTRY_ORDER)); height = .24
    for i, (frame, name, color) in enumerate(zip(frames, COHORTS, COLORS)):
        ax.barh(y + (i - 1) * height, pct(frame, "country_group", COUNTRY_ORDER),
                height, label=name, color=color)
    ax.set_yticks(y); ax.set_yticklabels(COUNTRY_ORDER); ax.invert_yaxis()
    ax.set_xlabel("Participants (%)"); ax.set_title("C. Country composition")
    ax.grid(axis="x", color="#dddddd", linewidth=.7); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(5); offsets = [-.22, 0, .22]
    for off, frame, name, color in zip(offsets, frames, COHORTS, COLORS):
        means = frame[CODES].mean().to_numpy()
        cis = 1.96 * frame[CODES].std(ddof=1).to_numpy() / np.sqrt(len(frame))
        ax.errorbar(x + off, means, yerr=cis, fmt="o", markersize=7,
                    capsize=4, linewidth=2, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(TRAITS, rotation=15, ha="right")
    ax.set_ylabel("Mean human IPIP score (1–5)")
    ax.set_title("D. OCEAN means and 95% confidence intervals")
    ax.legend(frameon=False, fontsize=9); style_axis(ax)
    ax.set_ylim(2.5, 4.2)

    fig.suptitle("Human IPIP cohorts used in the experiments", fontsize=18)
    path = os.path.join(OUT, "dataset_statistics_comparison.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_atlas(frames):
    """Compact distribution atlas: all demographics plus every OCEAN domain."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)

    ax = axes[0, 0]
    sizes = [len(frame) for frame in frames]
    bars = ax.bar(SHORT_LABELS, sizes, color=COLORS)
    ax.set_yscale("log"); ax.set_ylabel("Participants (log scale)")
    ax.set_title("Cohort sizes")
    for bar, value in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08,
                f"n={value:,}", ha="center", va="bottom", fontsize=9)
    style_axis(ax)

    ax = axes[0, 1]
    x = np.arange(len(AGE_ORDER))
    for i, (frame, color) in enumerate(zip(frames, COLORS)):
        ax.plot(x, pct(frame, "age_group", AGE_ORDER).to_numpy(), marker="o", linewidth=2.4,
                markersize=6, color=color, label=COHORTS[i])
    ax.set_xticks(x); ax.set_xticklabels(AGE_ORDER); ax.set_ylabel("Participants (%)")
    ax.set_title("Age composition"); style_axis(ax)

    ax = axes[0, 2]
    y = np.arange(len(COUNTRY_ORDER))
    for i, (frame, color) in enumerate(zip(frames, COLORS)):
        ax.plot(pct(frame, "country_group", COUNTRY_ORDER).to_numpy(), y, marker="o",
                linewidth=2.2, markersize=5, color=color, label=COHORTS[i])
    ax.set_yticks(y); ax.set_yticklabels(COUNTRY_ORDER); ax.invert_yaxis()
    ax.set_xlabel("Participants (%)"); ax.set_title("Country composition")
    ax.grid(axis="x", color="#dddddd", linewidth=.7); ax.set_axisbelow(True)

    for ax, code, trait in zip(axes[1].tolist() + [axes[0, 3]], CODES, TRAITS):
        if code == "N":
            ax.clear()
        positions = np.arange(3)
        arrays = [frame[code].dropna().to_numpy() for frame in frames]
        for position, values, color in zip(positions, arrays, COLORS):
            bp = ax.boxplot([values], positions=[position], widths=.58,
                            patch_artist=True, showfliers=False,
                            medianprops={"color": "black", "linewidth": 1.5})
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(.7)
        ax.set_xticks(positions); ax.set_xticklabels(SHORT_LABELS)
        ax.set_ylim(1, 5); ax.set_title(trait)
        if code in ("O", "N"):
            ax.set_ylabel("Human IPIP score (1–5)")
        style_axis(ax)

    handles = [Line2D([0], [0], color=c, marker="o", lw=3,
                      label=f"{name} (n={len(frame):,})")
               for c, name, frame in zip(COLORS, COHORTS, frames)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.04),
               ncol=3, frameon=False)
    fig.suptitle("Dataset distribution atlas", fontsize=18, y=1.08)
    path = os.path.join(OUT, "dataset_statistics_distribution_atlas.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(frames):
    rows = []
    for frame, name in zip(frames, COHORTS):
        record = {"cohort": name, "n": len(frame),
                  "countries_n": int(frame.country.nunique())}
        for code in CODES:
            record[f"{code}_mean"] = frame[code].mean()
            record[f"{code}_sd"] = frame[code].std(ddof=1)
        rows.append(record)
    path = os.path.join(OUT, "dataset_statistics_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = load_frames()
    outputs = [plot_dashboard(frames), plot_comparison(frames), plot_atlas(frames),
               write_summary(frames)]
    for path in outputs:
        print("wrote", path)


if __name__ == "__main__":
    main()
