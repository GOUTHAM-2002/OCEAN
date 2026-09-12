"""
Build OCEAN x OCEAN sampling-covariance matrices for the Mixtral variants and
plot heatmaps in the EXACT style of g_plots/correlation/ocean_corr_heatmap.png.
Profiles are z-scored within (variant, instrument) before pooling so instrument
mean differences don't create spurious correlations.
Also reports congruence with the human meta-analytic matrix (van der Linden 2010).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "g_plots", "correlation")
CODES = ["O", "C", "E", "A", "N"]
TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

HUMAN = np.array([   # van der Linden et al. 2010, corrected rho
    [1.00, .20, .43, .21, -.17],
    [ .20, 1.00, .29, .43, -.43],
    [ .43, .29, 1.00, .26, -.36],
    [ .21, .43, .26, 1.00, -.36],
    [-.17, -.43, -.36, -.36, 1.00]])

def corr_matrix(csv):
    df = pd.read_csv(csv)
    parts = []
    for inst, g in df.groupby("instrument"):
        z = g[CODES].copy()
        for c in CODES:
            s = z[c].std(ddof=1)
            z[c] = (z[c] - z[c].mean()) / (s if s > 1e-9 else 1.0)
        parts.append(z)
    pooled = pd.concat(parts, ignore_index=True)
    return np.corrcoef(pooled[CODES].values.T), len(pooled)

def plot_heatmap(M, fname):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="Oranges", vmin=-0.5, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85); cbar.set_label("Correlation")
    for i in range(5):
        for j in range(5):
            v = M[i, j]
            txt = "1" if i == j else f"{v:+.2f}".replace("+0.", ".").replace("-0.", "−.")
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                    fontweight="bold" if i == j else "normal", color=color)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(TRAITS, rotation=25, ha="right"); ax.set_yticklabels(TRAITS)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=150); plt.close()

def offdiag(M):
    iu = np.triu_indices(5, 1)
    return M[iu]

print(f"{'variant':12s} {'n':>4s}  congruence_with_human_r   offdiag pairs")
for label, csv, fname in [("baseline", "profiles_baseline.csv", "model_corr_baseline.png"),
                          ("HIGH b=.01", "profiles_high001.csv", "model_corr_high_beta0.01.png"),
                          ("LOW b=.01",  "profiles_low001.csv",  "model_corr_low_beta0.01.png")]:
    if not os.path.exists(os.path.join(HERE, csv)):
        print(f"{label}: missing {csv}"); continue
    M, n = corr_matrix(os.path.join(HERE, csv))
    plot_heatmap(M, fname)
    r = np.corrcoef(offdiag(M), offdiag(HUMAN))[0, 1]
    np.savetxt(os.path.join(OUT, fname.replace(".png", ".csv")), M, fmt="%.3f", delimiter=",")
    print(f"{label:12s} {n:4d}  r(model_offdiag, human_offdiag) = {r:+.3f}")
    print("   matrix:")
    for i, c in enumerate(CODES):
        print("    " + c + " " + " ".join(f"{M[i,j]:+.2f}" for j in range(5)))
print("plots ->", OUT)
