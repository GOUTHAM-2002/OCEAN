"""Per-beta correlation plots for the WizardLM-13B-V1.2 agent experiment.

For each beta and each agent direction (HIGH/LOW) we correlate "was DPO applied"
(0=baseline, 1=DPO adapter at that beta) against the OCEAN trait score, across the
10 per-cell scores (5 traits x 2 questionnaires). Spearman rho then answers: does
DPO with this agent move the traits, and in which direction?

  HIGH agent -> expect rho > 0 (DPO raises traits)
  LOW  agent -> expect rho < 0 (DPO lowers traits)

Each plot is a paired before->after view: baseline cells on the left, DPO cells on
the right, one line per (trait, questionnaire) cell, with rho and p annotated.

Outputs: g_plots/WizardLM/WizardLM-13B-V1.2/AGENT_{HIGH,LOW}/agent_{high,low}_beta<b>_corr.png
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "WizardLM-13B-V1.2"
OUTD = os.path.join(HERE, "g_plots", "WizardLM", SHORT)
TESTS = ["BFI-2", "FFPI"]
CODES = ["O", "C", "E", "A", "N"]
NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
         "A": "Agreeableness", "N": "Neuroticism"}
BETAS = ["0.01", "0.1", "0.5"]
TRAIT_COLORS = {"O": "#4C72B0", "C": "#55A868", "E": "#C44E52",
                "A": "#8172B3", "N": "#CCB974"}
TEST_MARKER = {"BFI-2": "o", "FFPI": "s"}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = np.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = np.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None, None
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * np.sqrt((n - 2) / (1 - r * r))
    # two-sided p via normal approx (same convention as downstream_stats_traits.py)
    p = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(t) / np.sqrt(2.0))))
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


def load(label):
    p = os.path.join(HERE, f"results20_{SHORT}_{label}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None


def cells(table):
    """Return list of (trait, test, score) for the 10 per-cell scores."""
    out = []
    for t in TESTS:
        for c in CODES:
            out.append((c, t, table[t][c][0]))
    return out


def make_plot(direction, folder, beta):
    base = load("baseline")
    dpo = load(f"{direction}_beta{beta}")
    if base is None or dpo is None:
        print(f"[{direction} beta{beta}] missing results, skipping")
        return
    base_cells = cells(base)
    dpo_cells = cells(dpo)

    # 20 (condition, score) points for the correlation
    conds = [0] * len(base_cells) + [1] * len(dpo_cells)
    scores = [s for _, _, s in base_cells] + [s for _, _, s in dpo_cells]
    rho, p = spearman(conds, scores)

    od = os.path.join(OUTD, folder)
    os.makedirs(od, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    for (c, t, sb), (_, _, sd) in zip(base_cells, dpo_cells):
        ax.plot([0, 1], [sb, sd], color=TRAIT_COLORS[c], alpha=0.6, lw=1.2, zorder=1)
        ax.scatter([0], [sb], color=TRAIT_COLORS[c], marker=TEST_MARKER[t],
                   s=45, edgecolor="black", linewidth=0.4, zorder=2)
        ax.scatter([1], [sd], color=TRAIT_COLORS[c], marker=TEST_MARKER[t],
                   s=45, edgecolor="black", linewidth=0.4, zorder=2)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline\n(no DPO)", f"DPO {direction.upper()}\n(β={beta})"])
    ax.set_ylabel("OCEAN score (1–5)")
    ax.set_ylim(1, 5)
    rho_s = f"{rho:+.3f}" if rho is not None else "n/a"
    p_s = f"{p:.3g}" if p is not None else "n/a"
    ax.set_title(f"{SHORT} — {direction.upper()} agent, β={beta}\n"
                 f"Spearman ρ(DPO applied, trait score) = {rho_s}  (p = {p_s}, n=20)")

    trait_handles = [plt.Line2D([], [], color=TRAIT_COLORS[c], marker="o", lw=0,
                                markeredgecolor="black", markeredgewidth=0.4, label=NAMES[c])
                     for c in CODES]
    test_handles = [plt.Line2D([], [], color="#666666", marker=TEST_MARKER[t], lw=0,
                               markeredgecolor="black", markeredgewidth=0.4, label=t)
                    for t in TESTS]
    leg1 = ax.legend(handles=trait_handles, loc="upper left", fontsize=8, title="Trait")
    ax.add_artist(leg1)
    ax.legend(handles=test_handles, loc="lower right", fontsize=8, title="Questionnaire")

    plt.tight_layout()
    out = os.path.join(od, f"agent_{direction}_beta{beta}_corr.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[{direction} beta{beta}] rho={rho_s} p={p_s} -> {out}")


if __name__ == "__main__":
    print(f"{'direction':10s} {'beta':6s} {'spearman_rho':>13s} {'p':>10s}")
    for direction, folder in [("high", "AGENT_HIGH"), ("low", "AGENT_LOW")]:
        for beta in BETAS:
            make_plot(direction, folder, beta)
    print("AGENT_CORR_PLOTS_DONE")
