"""Per-beta personality-transfer correlation plots for the 1000-user experiment
(Nous-Hermes-2-Mixtral-8x7B-SFT).

For each beta, and each OCEAN trait, we correlate ACROSS THE 1000 USERS:
  x = the annotator's own IPIP-300 trait score
  y = the model's DPO-induced shift in that trait,
      Δ = (DPO score - baseline score), averaged over BFI-2 and FFPI
with PEARSON r. A positive r means: users who score high in a trait push the
DPO'd model higher in that same trait -> DPO transfers the annotator's personality.

This is the 1000-user analog of user_corr_plots.py, reporting Pearson (not Spearman).

Outputs (one figure per beta, 5 trait subplots each, up to 1000 points each):
  g_plots/Nous-Hermes-2-Mixtral-8x7B-SFT/USERS_1000_CORR/users1000_beta<b>_corr.png
plus a summary CSV of r/p for all (beta x trait) cells.
"""
import os, json, csv, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Publication-scale typography: remains legible when the full figure is reduced
# to a Nature-style page width.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "Nous-Hermes-2-Mixtral-8x7B-SFT"
OUTD = os.path.join(HERE, "g_plots", SHORT, "USERS_1000_CORR")
TESTS = ["BFI-2", "FFPI"]
CODES = ["O", "C", "E", "A", "N"]
NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
         "A": "Agreeableness", "N": "Neuroticism"}
SCORE_COL = {"O": "Openness_score", "C": "Conscientiousness_score",
             "E": "Extraversion_score", "A": "Agreeableness_score",
             "N": "Neuroticism_score"}
IPIP_ITEMS_PER_DOMAIN = 60  # IPIP-300: 60 items/domain, so raw 60-300 -> /60 = 1-5 mean
BETAS = ["0.01"]
TRAIT_COLORS = {"O": "#4C72B0", "C": "#55A868", "E": "#C44E52",
                "A": "#8172B3", "N": "#CCB974"}


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
    from math import erf, sqrt
    t = r * np.sqrt((n - 2) / (1 - r * r))
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
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


def table(label):
    p = os.path.join(HERE, f"results_{SHORT}_{label}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None


def trait_means(tab):
    """{trait: mean over BFI-2 & FFPI}."""
    return {c: float(np.mean([tab[t][c][0] for t in TESTS])) for c in CODES}


def load_annotator_scores():
    """case_id -> {trait: annotator IPIP-300 score}. Skips non-numeric ids
    (any synthetic AGENT_LOW/AGENT_HIGH rows)."""
    out = {}
    with open(os.path.join(HERE, "ipip300_responses_and_scores_10000.csv")) as fh:
        for row in csv.DictReader(fh):
            try:
                cid = int(row["case_id"])
            except (ValueError, TypeError):
                continue
            out[cid] = {c: float(row[SCORE_COL[c]]) / IPIP_ITEMS_PER_DOMAIN for c in CODES}
    return out


def user_ids():
    return json.load(open(os.path.join(HERE, "user_ids_1000.json")))


def main():
    os.makedirs(OUTD, exist_ok=True)
    annot = load_annotator_scores()
    base = trait_means(table("baseline"))
    users = user_ids()

    summary = []  # (beta, trait, n, r, p)
    for beta in BETAS:
        # gather per-user (annotator score, model delta) for each trait
        data = {c: {"x": [], "y": []} for c in CODES}
        n_used = 0
        for u in users:
            tab = table(f"user_{u}_beta{beta}")
            if tab is None or u not in annot:
                continue
            tm = trait_means(tab)
            n_used += 1
            for c in CODES:
                data[c]["x"].append(annot[u][c])
                data[c]["y"].append(tm[c] - base[c])

        # 3 on top, 2 centered below: 6-col grid, each panel spans 2 cols,
        # bottom row offset by 1 col to centre it.
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.75)
        slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],   # O, C, E
                 gs[1, 1:3], gs[1, 3:5]]                # A, N (centred)
        axes = [fig.add_subplot(s) for s in slots]
        # 1000-user AGGREGATE DPO (one model trained on data pooled from all
        # 1000 users): highlighted so it is never mistaken for a per-user point
        agg_tab = table(f"ALLUSERS1000_beta{beta}")
        agg_tm = trait_means(agg_tab) if agg_tab is not None else None
        for ax, c in zip(axes, CODES):
            xs, ys = data[c]["x"], data[c]["y"]
            r, p = pearson(xs, ys)
            ax.scatter(xs, ys, s=26, color=TRAIT_COLORS[c], alpha=0.7,
                       edgecolor="black", linewidth=0.3)
            if agg_tm is not None and xs:
                x_agg = float(np.mean(xs))  # x = mean annotator trait of cohort
                y_agg = agg_tm[c] - base[c]
                ax.scatter([x_agg], [y_agg], marker="*", s=420, color="gold",
                           edgecolor="black", linewidth=1.3, zorder=6,
                           label="1000-user aggregate DPO")
                ax.legend(loc="best", fontsize=7, frameon=True)
            ax.axhline(0, color="black", linewidth=0.7)
            # light OLS trend line for visual guidance
            if len(xs) >= 2 and np.std(xs) > 0:
                b1, b0 = np.polyfit(xs, ys, 1)
                xr = np.array([min(xs), max(xs)])
                ax.plot(xr, b0 + b1 * xr, color=TRAIT_COLORS[c], lw=1.5, ls="--", alpha=0.9)
            r_s = f"{r:+.3f}" if r is not None else "n/a"
            p_s = ("< 1e-16" if (p is not None and p < 1e-16)
                   else f"{p:.3g}" if p is not None else "n/a")
            ax.set_title(f"{NAMES[c]}\nPearson r = {r_s}  (p = {p_s})")
            ax.set_xlabel(f"Annotator {NAMES[c]} (IPIP-300, 1–5)")
            summary.append((beta, c, len(xs), r, p))
        ylab = f"Model trait shift Δ vs baseline\n(BFI-2 & FFPI mean, β={beta})"
        for i in (0, 3):  # leftmost panel of each row
            axes[i].set_ylabel(ylab)
        fig.suptitle(f"{SHORT} — 1000-user personality transfer, β={beta}  "
                     f"(n={n_used} users)   x = annotator trait, y = DPO-induced shift",
                     fontsize=17)
        out = os.path.join(OUTD, f"users1000_beta{beta}_corr.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[beta{beta}] n={n_used} users -> {out}")

    # summary CSV
    csv_path = os.path.join(OUTD, "users1000_transfer_correlations.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beta", "trait", "n_users", "pearson_r", "pearson_p"])
        for beta, c, n, r, p in summary:
            w.writerow([beta, NAMES[c], n,
                        f"{r:.4f}" if r is not None else "",
                        f"{p:.4g}" if p is not None else ""])
    print(f"[summary] wrote {csv_path}")
    print("\nbeta  trait              n    r        p")
    for beta, c, n, r, p in summary:
        rr = f"{r:+.3f}" if r is not None else "  n/a "
        pp = f"{p:.3g}" if p is not None else "n/a"
        print(f"{beta:5s} {NAMES[c]:17s} {n:4d}  {rr}  {pp}")
    print("USER_CORR_PLOTS_1000_DONE")


if __name__ == "__main__":
    main()
