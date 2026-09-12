"""Prove sycophancy and the Agreeableness trait share a direction/subspace in
WizardLM-13B-V1.2. Uses the per-example difference vectors from
extract_syc_traits.py. Produces cosine comparisons + a PCA projection figure.
Output -> g_plots/mech_exp/sycophancy_agreeableness/.
"""
import os, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "g_plots/mech_exp/sycophancy_agreeableness"; os.makedirs(OUT, exist_ok=True)
CODES = ["O", "C", "E", "A", "N"]
NM = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness", "N": "Neuroticism"}
LIDX = 15   # hidden_states index = output of decoder layer 14 (the causal/representation layer)

blob = torch.load("syc_traits_diffs.pt")
SYC = blob["syc_diffs"].numpy()                 # [N_syc, L+1, H]
TD = {c: blob["trait_diffs"][c].numpy() for c in CODES}   # each [60, L+1, H]
nlayers = SYC.shape[1]

def unit(v): return v / (np.linalg.norm(v) + 1e-9)
def cos(a, b): return float(np.dot(unit(a), unit(b)))

syc_mean = SYC.mean(0)                            # [L+1, H]
trait_mean = {c: TD[c].mean(0) for c in CODES}    # [L+1, H]

# ---- 1. cosine(sycophancy, trait) at the representation layer ----
print(f"=== cosine(sycophancy direction, trait direction) at layer 14 ===")
cos14 = {c: cos(syc_mean[LIDX], trait_mean[c][LIDX]) for c in CODES}
for c in CODES:
    print(f"  {NM[c]:18s}: {cos14[c]:+.3f}")
# control: average pairwise cosine among the 5 traits (is A-syc special?)
pair = [cos(trait_mean[a][LIDX], trait_mean[b][LIDX]) for i, a in enumerate(CODES) for b in CODES[i+1:]]
print(f"  [control] mean |cos| among trait pairs: {np.mean(np.abs(pair)):.3f}")
print(f"  [control] cos(sycophancy, random dir): {np.mean([cos(syc_mean[LIDX], np.random.randn(SYC.shape[2])) for _ in range(200)]):+.3f}")

# ---- 2. layer-wise cosine syc vs Agreeableness and vs Neuroticism ----
layers = list(range(1, nlayers))
cosA = [cos(syc_mean[l], trait_mean["A"][l]) for l in layers]
cosN = [cos(syc_mean[l], trait_mean["N"][l]) for l in layers]
cosmean_others = [np.mean([cos(syc_mean[l], trait_mean[c][l]) for c in ["O", "C", "E"]]) for l in layers]

# ---- 3. projection: how much of sycophancy lies along each trait axis ----
print(f"\n=== sycophancy mean direction projected onto each unit trait axis (layer 14) ===")
proj = {c: float(np.dot(syc_mean[LIDX], unit(trait_mean[c][LIDX]))) for c in CODES}
for c in CODES: print(f"  {NM[c]:18s}: {proj[c]:+.2f}")

# ---- 4. PCA on combined sycophancy + Agreeableness + Neuroticism diff clouds ----
def L14(x): return x[:, LIDX, :]
X_syc, X_A, X_N = L14(SYC), L14(TD["A"]), L14(TD["N"])
Xall = np.concatenate([X_syc, X_A, X_N], 0)
mu = Xall.mean(0)
U, S, Vt = np.linalg.svd(Xall - mu, full_matrices=False)
PC = Vt[:2]                                       # top-2 principal components
def pcs(x): return (x - mu) @ PC.T
P_syc, P_A, P_N = pcs(X_syc), pcs(X_A), pcs(X_N)
# cosine between the principal axis of syc-cloud and A-cloud
def pc1(x):
    u, s, vt = np.linalg.svd(x - x.mean(0), full_matrices=False); return vt[0]
print(f"\n=== PCA ===")
print(f"  cos(PC1 of sycophancy cloud, PC1 of Agreeableness cloud): {abs(cos(pc1(X_syc), pc1(X_A))):.3f}")
print(f"  cos(PC1 of sycophancy cloud, PC1 of Neuroticism  cloud): {abs(cos(pc1(X_syc), pc1(X_N))):.3f}")
print(f"  centroid distance syc<->A: {np.linalg.norm(P_syc.mean(0)-P_A.mean(0)):.2f} | "
      f"syc<->N: {np.linalg.norm(P_syc.mean(0)-P_N.mean(0)):.2f}")

# ---- figures ----
# Fig 1: cosine bar
fig, ax = plt.subplots(figsize=(7, 4))
cols = ["#C44E52" if c == "A" else "#bbbbbb" for c in CODES]
ax.bar([NM[c][:5] for c in CODES], [cos14[c] for c in CODES], color=cols, edgecolor="black")
ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("cosine with sycophancy direction")
ax.set_title("Sycophancy direction aligns with Agreeableness (layer 14)")
plt.tight_layout(); plt.savefig(f"{OUT}/cosine_bar.png", dpi=150); plt.close()

# Fig 2: PCA scatter
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(P_A[:, 0], P_A[:, 1], s=22, alpha=.7, label="Agreeableness (high−low)", color="#C44E52")
ax.scatter(P_syc[:, 0], P_syc[:, 1], s=14, alpha=.5, label="Sycophancy (matches user−honest)", color="#4C72B0")
ax.scatter(P_N[:, 0], P_N[:, 1], s=22, alpha=.7, label="Neuroticism (high−low) [control]", color="#55A868")
for P, c, lab in [(P_A, "#C44E52", "A"), (P_syc, "#4C72B0", "syc"), (P_N, "#55A868", "N")]:
    ax.scatter(*P.mean(0), s=320, marker="*", color=c, edgecolor="black", zorder=5)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8)
ax.set_title("Per-example difference vectors in shared PCA space (layer 14)\nstars = centroids")
plt.tight_layout(); plt.savefig(f"{OUT}/pca_scatter.png", dpi=150); plt.close()

# Fig 3: layer curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(layers, cosA, marker="o", ms=3, label="cos(syc, Agreeableness)", color="#C44E52")
ax.plot(layers, cosmean_others, marker="s", ms=3, label="cos(syc, O/C/E avg)", color="#999999")
ax.plot(layers, cosN, marker="^", ms=3, label="cos(syc, Neuroticism)", color="#55A868")
ax.axhline(0, color="black", lw=0.8); ax.set_xlabel("layer"); ax.set_ylabel("cosine")
ax.set_title("Sycophancy tracks Agreeableness across depth"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/layer_cosine.png", dpi=150); plt.close()

json.dump({"cos14": cos14, "proj": proj}, open(f"{OUT}/summary.json", "w"), indent=2)
print(f"\nfigures -> {OUT}")
