"""The Last Dance — sex violins ONLY. Individual per-person OCEAN scores (NOT
averaged across people). Shows DPO-on-Male vs DPO-on-Female:
  - per instrument (BFI-2, FFPI) with the baseline model as a reference line
  - broken across demographics (age brackets, country groups)
Each violin = 100 (or per-cell) individual people; bold marker = group mean +/-95% CI;
p = t-test Male vs Female.
"""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sps

OUT = "g_plots/the_last_dance"; os.makedirs(OUT, exist_ok=True)
CODES = ["O","C","E","A","N"]; NAMES = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
BLUE = "#4C72B0"; RED = "#C44E52"
d = pd.read_csv("ipip300_responses_and_scores.csv"); d["case_id"] = d["case_id"].astype(str)
ENG = {"Canada","UK","Australia","Ireland","New Zealand"}
d["cgrp"] = np.where(d.country=="USA","USA",np.where(d.country.isin(ENG),"OtherEng","Rest"))
d["aband"] = pd.cut(d.age,[0,18,25,35,200],right=False,labels=["<18","18-24","25-34","35+"]).astype(str)

base = json.load(open("results20_WizardLM-13B-V1.2_baseline.json"))["table"]
def baseline(inst, c): return base[inst][c][0]
def baseline_comb(c): return np.mean([base[t][c][0] for t in ["BFI-2","FFPI"]])

def load(cid):
    p = f"results_lastdance/{cid}.json"
    if not os.path.exists(p): return None
    return json.load(open(p))["table"]
rows = []
for cid in set(map(str, json.load(open("ld_males.json"))+json.load(open("ld_females.json")))):
    t = load(cid); r = d[d.case_id==cid]
    if t is None or len(r)==0: continue
    r = r.iloc[0]; rec = dict(sex=r.sex, cgrp=r.cgrp, aband=r.aband)
    for c in CODES:
        rec[f"BFI2_{c}"] = t["BFI-2"][c][0]; rec[f"FFPI_{c}"] = t["FFPI"][c][0]
        rec[f"COMB_{c}"] = np.mean([t["BFI-2"][c][0], t["FFPI"][c][0]])
    rows.append(rec)
df = pd.DataFrame(rows)
print(f"n={len(df)} ({(df.sex=='male').sum()} M, {(df.sex=='female').sum()} F)")

def mean_ci(v):
    v = np.asarray(v); m = v.mean(); ci = 1.96*v.std(ddof=1)/np.sqrt(len(v)); return m, ci

def draw_pair(ax, xc, mvals, fvals, w=0.34):
    for vals, off, col in [(mvals,-0.2,BLUE),(fvals,0.2,RED)]:
        vp = ax.violinplot(vals,[xc+off],widths=w,showextrema=False)
        for b in vp['bodies']: b.set_facecolor(col); b.set_alpha(.55)
        m,ci = mean_ci(vals)
        ax.errorbar(xc+off, m, yerr=ci, fmt='o', color=col, ms=6, capsize=4, mec='black', zorder=5)

# ===== per-instrument overall figures (Male vs Female), with baseline line =====
for inst, key in [("BFI-2","BFI2"),("FFPI","FFPI")]:
    fig, ax = plt.subplots(figsize=(13,6)); x = np.arange(5)
    M = df[df.sex=="male"]; F = df[df.sex=="female"]
    for i,c in enumerate(CODES):
        draw_pair(ax, i, M[f"{key}_{c}"].values, F[f"{key}_{c}"].values)
        ax.plot([i-0.45,i+0.45],[baseline(inst,c)]*2, ls="--", color="black", lw=1.6)
        _,p = sps.ttest_ind(M[f"{key}_{c}"], F[f"{key}_{c}"])
        top = max(M[f"{key}_{c}"].max(), F[f"{key}_{c}"].max())
        ax.text(i, top+0.05, f"p={p:.3f}", ha="center", fontsize=9, fontweight="bold" if p<0.05 else "normal")
    ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel(f"{inst} score (1-5), individual people")
    ax.set_title(f"DPO on Male vs Female — {inst} (each violin = individual people; dashed = baseline model)")
    ax.legend([plt.Rectangle((0,0),1,1,fc=BLUE,alpha=.6), plt.Rectangle((0,0),1,1,fc=RED,alpha=.6),
               plt.Line2D([0],[0],ls="--",color="black")], ["Male-DPO","Female-DPO","baseline"], loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUT}/sex_violin_{key}.png",dpi=150); plt.close()

# ===== across demographics: one panel per trait, x = demographic bracket =====
def demo_figure(col, order, fname, label):
    fig, axes = plt.subplots(1,5,figsize=(22,7),sharey=False)
    for ax,c,name in zip(axes,CODES,NAMES):
        xs = [g for g in order if ((df[col]==g)&(df.sex=='male')).sum()>=6 and ((df[col]==g)&(df.sex=='female')).sum()>=6]
        for xi,g in enumerate(xs):
            M = df[(df[col]==g)&(df.sex=='male')][f"COMB_{c}"].values
            F = df[(df[col]==g)&(df.sex=='female')][f"COMB_{c}"].values
            draw_pair(ax, xi, M, F)
            _,p = sps.ttest_ind(M,F)
            ax.text(xi, max(M.max(),F.max())+0.04, ("*" if p<0.05 else ""), ha="center", fontsize=14, color="green")
        ax.plot([-0.5,len(xs)-0.5],[baseline_comb(c)]*2, ls="--", color="black", lw=1.2)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{g}\n(n={((df[col]==g)&(df.sex=='male')).sum()}|{((df[col]==g)&(df.sex=='female')).sum()})" for g in xs], fontsize=8)
        ax.set_title(name, fontsize=11)
    axes[0].set_ylabel("OCEAN score (1-5), individual people")
    fig.suptitle(f"DPO on Male (blue) vs Female (red) across {label}   (* = M vs F p<.05; dashed = baseline)", fontsize=13)
    plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(f"{OUT}/{fname}",dpi=150); plt.close()

demo_figure("aband", ["<18","18-24","25-34","35+"], "sex_violin_by_age.png", "AGE brackets")
demo_figure("cgrp", ["USA","OtherEng","Rest"], "sex_violin_by_country.png", "COUNTRY groups")
print("wrote sex_violin_BFI2 / FFPI / by_age / by_country ->", OUT)
