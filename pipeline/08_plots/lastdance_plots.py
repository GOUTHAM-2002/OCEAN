"""The Last Dance figures -> g_plots/the_last_dance/.
Shows: (1) violin plots of the model's OCEAN per trait, Male vs Female DPO;
(2) human vs model sex-gap per trait (the flattening / bias);
(3) fidelity = model gap / human gap per trait.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sps

OUT = "g_plots/the_last_dance"; os.makedirs(OUT, exist_ok=True)
CODES = ["O", "C", "E", "A", "N"]; NAMES = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
TESTS = ["BFI-2", "FFPI"]
males = set(map(str, json.load(open("ld_males.json")))); females = set(map(str, json.load(open("ld_females.json"))))

def load(cid):
    p = f"results_lastdance/{cid}.json"
    if not os.path.exists(p): return None
    t = json.load(open(p))["table"]
    return {c: np.mean([t[ti][c][0] for ti in TESTS]) for c in CODES}

mvals = {c: [] for c in CODES}; fvals = {c: [] for c in CODES}
nm = nf = 0
for cid in males:
    v = load(cid)
    if v: nm += 1; [mvals[c].append(v[c]) for c in CODES]
for cid in females:
    v = load(cid)
    if v: nf += 1; [fvals[c].append(v[c]) for c in CODES]
print(f"loaded model OCEAN: {nm} male, {nf} female")

# human OCEAN for the same people (ground truth)
d = pd.read_csv("ipip300_responses_and_scores.csv"); d["case_id"] = d["case_id"].astype(str)
SC = {"O":"Openness_score","C":"Conscientiousness_score","E":"Extraversion_score","A":"Agreeableness_score","N":"Neuroticism_score"}
hum = {sex: {c: [float(d.loc[d.case_id == cid, SC[c]].iloc[0])/60 for cid in ids if (d.case_id == cid).any()] for c in CODES}
       for sex, ids in [("M", males), ("F", females)]}

# ---- (1) VIOLIN: model OCEAN, M vs F, per trait ----
fig, ax = plt.subplots(figsize=(12, 6)); pos = np.arange(len(CODES))
for i, c in enumerate(CODES):
    vpM = ax.violinplot(mvals[c], [i-0.2], widths=0.35, showmeans=True)
    vpF = ax.violinplot(fvals[c], [i+0.2], widths=0.35, showmeans=True)
    for b in vpM['bodies']: b.set_facecolor("#4C72B0"); b.set_alpha(.7)
    for b in vpF['bodies']: b.set_facecolor("#C44E52"); b.set_alpha(.7)
ax.set_xticks(pos); ax.set_xticklabels(NAMES); ax.set_ylabel("Model OCEAN score (1-5)")
ax.set_title("WizardLM-13B personality after DPO on 100 male vs 100 female (matched on country & age)")
ax.legend([plt.Rectangle((0,0),1,1,fc="#4C72B0"), plt.Rectangle((0,0),1,1,fc="#C44E52")], ["Male-DPO","Female-DPO"])
plt.tight_layout(); plt.savefig(f"{OUT}/violin_model_ocean.png", dpi=150); plt.close()

# ---- (2) human gap vs model gap per trait ----
rows = []
for c in CODES:
    hg = np.mean(hum["F"][c]) - np.mean(hum["M"][c])         # human F-M gap
    mg = np.mean(fvals[c]) - np.mean(mvals[c])               # model F-M gap
    t, pval = sps.ttest_ind(fvals[c], mvals[c])
    rows.append((c, hg, mg, pval))
fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(CODES)); w = 0.38
ax.bar(x-w/2, [r[1] for r in rows], w, label="HUMAN F−M gap (real)", color="#888888", edgecolor="black")
ax.bar(x+w/2, [r[2] for r in rows], w, label="MODEL F−M gap (after DPO)", color="#C44E52", edgecolor="black")
for i,(c,hg,mg,pv) in enumerate(rows):
    if pv < 0.05: ax.annotate("*", (i+w/2, mg), ha="center", va="bottom" if mg>=0 else "top")
ax.axhline(0, color="black", lw=1); ax.set_xticks(x); ax.set_xticklabels(NAMES)
ax.set_ylabel("Female − Male difference"); ax.legend()
ax.set_title("The model flattens real sex differences (* = model gap p<.05)")
plt.tight_layout(); plt.savefig(f"{OUT}/human_vs_model_gap.png", dpi=150); plt.close()

# ---- (3) fidelity = model gap / human gap ----
fig, ax = plt.subplots(figsize=(9, 5))
fid = [ (r[2]/r[1] if abs(r[1])>1e-6 else 0) for r in rows ]
ax.bar(NAMES, fid, color=["#C44E52" if c in ("A","N") else "#4C72B0" for c in CODES], edgecolor="black")
ax.axhline(1, ls="--", color="gray", label="faithful (model=human)")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("fidelity = model gap / human gap"); ax.legend()
ax.set_title("How faithfully the model transmits each sex difference\n(low/negative = flattened or reversed)")
plt.tight_layout(); plt.savefig(f"{OUT}/fidelity.png", dpi=150); plt.close()

# ---- summary table ----
with open(f"{OUT}/summary.txt", "w") as fh:
    fh.write(f"n = {nm} male, {nf} female (matched on country x age)\n\n")
    fh.write(f"{'Trait':18s} | human F-M | model F-M | fidelity | p(model)\n")
    for (c,hg,mg,pv),f in zip(rows,fid):
        fh.write(f"{NAMES[CODES.index(c)]:18s} | {hg:+.3f}   | {mg:+.3f}   | {f:+.2f}   | {pv:.3f}\n")
print(open(f"{OUT}/summary.txt").read())
print("figures ->", OUT)
