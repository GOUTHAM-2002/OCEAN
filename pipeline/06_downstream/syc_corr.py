"""Behavioral test of the hypothesis: does DPO's sycophancy increase explain the
Agreeableness rise? Correlate, across the 20 user adapters, each adapter's
sycophancy-propensity shift vs its shift in EACH OCEAN trait. Whichever trait
co-moves with sycophancy is the one sycophancy is behaviorally tied to.
"""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SHORT = "WizardLM-13B-V1.2"; TESTS = ["BFI-2", "FFPI"]; CODES = ["O", "C", "E", "A", "N"]
NM = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness", "N": "Neuroticism"}
OUT = "g_plots/mech_exp/sycophancy_agreeableness"; os.makedirs(OUT, exist_ok=True)
USERS = json.load(open("user_ids.json"))

prop = {}
for f in glob.glob("syc_prop_gpu*.json"):
    prop.update(json.load(open(f)))
base_syc = prop["baseline"]

def tv(label):
    p = f"results20_{SHORT}_{label}.json"
    if not os.path.exists(p): return None
    t = json.load(open(p))["table"]
    return {c: np.mean([t[ti][c][0] for ti in TESTS]) for c in CODES}

base_tv = tv("baseline")
rows = []  # (user, dsyc, {trait:dtrait})
for u in USERS:
    lab = f"user_{u}_beta0.01"
    if lab not in prop: continue
    av = tv(lab)
    if av is None: continue
    rows.append((u, prop[lab] - base_syc, {c: av[c] - base_tv[c] for c in CODES}))

dsyc = np.array([r[1] for r in rows])
print(f"n={len(rows)} adapters | baseline sycophancy propensity = {base_syc:+.2f}")
print(f"mean sycophancy shift under DPO = {dsyc.mean():+.2f} (so DPO {'increases' if dsyc.mean()>0 else 'decreases'} sycophancy)\n")
print("Correlation across the 20 adapters: Δsycophancy vs Δtrait")
res = {}
for c in CODES:
    dt = np.array([r[2][c] for r in rows])
    r = float(np.corrcoef(dsyc, dt)[0, 1])
    res[c] = r
    print(f"  Δsycophancy vs Δ{NM[c]:18s}: r = {r:+.3f}")
# general factor
dgf = np.array([np.mean([r[2][c] for c in ['O','C','E','A']]) - r[2]['N'] for r in rows])
rgf = float(np.corrcoef(dsyc, dgf)[0,1])
print(f"  Δsycophancy vs Δ(general factor)   : r = {rgf:+.3f}")

# scatter: Δsyc vs ΔAgreeableness
dA = np.array([r[2]["A"] for r in rows])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(dsyc, dA, s=45, color="#C44E52", edgecolor="black")
b, a = np.polyfit(dsyc, dA, 1)
xs = np.array([dsyc.min(), dsyc.max()]); ax.plot(xs, a + b*xs, "--", color="black", lw=1.2)
ax.set_xlabel("Δ sycophancy propensity (adapter − baseline)")
ax.set_ylabel("Δ Agreeableness (adapter − baseline)")
ax.set_title(f"Per-adapter: sycophancy rise vs Agreeableness rise\nPearson r = {res['A']:+.2f}  (n={len(rows)})")
plt.tight_layout(); plt.savefig(f"{OUT}/behavioral_scatter.png", dpi=150); plt.close()

# bar of correlations
fig, ax = plt.subplots(figsize=(7.5, 4))
cs = [res[c] for c in CODES] + [rgf]
labs = [NM[c][:5] for c in CODES] + ["GenFac"]
cols = ["#C44E52" if c == "A" else "#999999" for c in CODES] + ["#4C72B0"]
ax.bar(labs, cs, color=cols, edgecolor="black"); ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Pearson r with Δsycophancy"); ax.set_ylim(-1, 1)
ax.set_title("Which trait's DPO shift co-moves with the sycophancy shift? (20 adapters)")
plt.tight_layout(); plt.savefig(f"{OUT}/behavioral_corr_bar.png", dpi=150); plt.close()
json.dump({"base_syc": base_syc, "mean_dsyc": float(dsyc.mean()), "corr": res, "corr_gf": rgf},
          open(f"{OUT}/behavioral_summary.json", "w"), indent=2)
print(f"\nfigures -> {OUT}")
