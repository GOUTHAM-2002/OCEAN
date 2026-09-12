"""The Last Dance — demographic plots. Bootstrap-mean violins (so small-but-real
group gaps are visible) for the sex contrast, plus a human-vs-model gap panel for
the most-different axes (sex, age teens-vs-adults, USA-vs-rest)."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sps

OUT = "g_plots/the_last_dance"; os.makedirs(OUT, exist_ok=True)
CODES = ["O","C","E","A","N"]; NAMES = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
TESTS = ["BFI-2","FFPI"]; rng = np.random.RandomState(0)
d = pd.read_csv("ipip300_responses_and_scores.csv"); d["case_id"] = d["case_id"].astype(str)
ENG = {"Canada","UK","Australia","Ireland","New Zealand"}
d["cgrp"] = np.where(d.country=="USA","USA",np.where(d.country.isin(ENG),"OtherEng","Rest"))
d["aband"] = pd.cut(d.age,[0,18,25,35,200],right=False,labels=["<18","18-24","25-34","35+"]).astype(str)
SC = {"O":"Openness_score","C":"Conscientiousness_score","E":"Extraversion_score","A":"Agreeableness_score","N":"Neuroticism_score"}
def mo(cid):
    p = f"results_lastdance/{cid}.json"
    if not os.path.exists(p): return None
    t = json.load(open(p))["table"]; return {c: np.mean([t[ti][c][0] for ti in TESTS]) for c in CODES}
rows = []
for cid in set(map(str, json.load(open("ld_males.json"))+json.load(open("ld_females.json")))):
    m = mo(cid); r = d[d.case_id==cid]
    if m is None or len(r)==0: continue
    r = r.iloc[0]
    rows.append(dict(sex=r.sex,cgrp=r.cgrp,aband=r.aband,**{f"m{c}":m[c] for c in CODES},**{f"h{c}":float(r[SC[c]])/60 for c in CODES}))
df = pd.DataFrame(rows)

def boot_means(vals, n=4000):
    vals = np.asarray(vals); return np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)])

# ---- (1) bootstrap-mean violins: Male vs Female, per trait ----
M = df[df.sex=="male"]; F = df[df.sex=="female"]
fig, ax = plt.subplots(figsize=(13,6)); x = np.arange(5)
for i,c in enumerate(CODES):
    bm = boot_means(M[f"m{c}"]); bf = boot_means(F[f"m{c}"])
    vM = ax.violinplot(bm,[i-0.22],widths=0.38,showextrema=False)
    vF = ax.violinplot(bf,[i+0.22],widths=0.38,showextrema=False)
    for b in vM['bodies']: b.set_facecolor("#4C72B0"); b.set_alpha(.85)
    for b in vF['bodies']: b.set_facecolor("#C44E52"); b.set_alpha(.85)
    _,p = sps.ttest_ind(M[f"m{c}"],F[f"m{c}"])
    ax.text(i, max(bm.max(),bf.max())+0.004, f"p={p:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("Model OCEAN — distribution of the GROUP MEAN (bootstrap)")
ax.set_title("Male-DPO vs Female-DPO: the group means DO separate (esp. Neuroticism, which reverses)")
ax.legend([plt.Rectangle((0,0),1,1,fc="#4C72B0"),plt.Rectangle((0,0),1,1,fc="#C44E52")],["Male-DPO","Female-DPO"])
plt.tight_layout(); plt.savefig(f"{OUT}/sex_bootstrap_violin.png",dpi=150); plt.close()

# ---- (2) human gap vs model gap, for the most-different axes ----
contrasts = [("Sex: F − M", df.sex=="female", df.sex=="male"),
             ("Age: teens(<18) − adults(35+)", df.aband=="<18", df.aband=="35+"),
             ("Country: USA − Rest", df.cgrp=="USA", df.cgrp=="Rest")]
fig, axes = plt.subplots(1,3,figsize=(16,5),sharey=True)
for ax,(title,ma,mb) in zip(axes,contrasts):
    A=df[ma]; B=df[mb]; x=np.arange(5); w=0.38
    hg=[A[f"h{c}"].mean()-B[f"h{c}"].mean() for c in CODES]
    mg=[A[f"m{c}"].mean()-B[f"m{c}"].mean() for c in CODES]
    ax.bar(x-w/2,hg,w,label="HUMAN gap",color="#888888",edgecolor="black")
    ax.bar(x+w/2,mg,w,label="MODEL gap",color="#C44E52",edgecolor="black")
    ax.axhline(0,color="black",lw=1); ax.set_xticks(x); ax.set_xticklabels([n[:5] for n in NAMES],rotation=30)
    ax.set_title(f"{title}\n(n={len(A)} vs {len(B)})",fontsize=10)
axes[0].set_ylabel("group difference (1-5 scale)"); axes[0].legend()
plt.tight_layout(rect=[0,0,1,0.90]); fig.suptitle("The model flattens every demographic difference (biggest erasure = age)",fontsize=13); plt.savefig(f"{OUT}/demographic_gaps.png",dpi=150); plt.close()
print("wrote sex_bootstrap_violin.png + demographic_gaps.png ->", OUT)
