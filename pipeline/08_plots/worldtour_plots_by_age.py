"""World Tour — Male vs Female DPO across 7 countries AND 3 age brackets. Same
violin style as worldtour_sex_by_country but x-axis = country x age (21 groups).
Massive figure. Uses BFI-2+FFPI when available. -> g_plots/the_last_dance/."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sps

# Preserve the existing plot design while making all text paper-legible.
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titleweight": "bold",
})

OUT = "g_plots/the_last_dance"
CODES=["O","C","E","A","N"]; NAMES=["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
BLUE="#4C72B0"; RED="#C44E52"
COUNTRIES=["USA","India","China","South Afric","Germany","Australia","Brazil"]; CLAB={"South Afric":"S.Afr"}
ABS=["18-24","25-34","35+"]
meta={int(k):v for k,v in json.load(open("worldtour_meta.json")).items()}
base=json.load(open("results20_WizardLM-13B-V1.2_baseline.json"))["table"]
have_ffpi=os.path.isdir("results_worldtour_ffpi") and len(os.listdir("results_worldtour_ffpi"))>500
def base_score(c):
    ts=["BFI-2","FFPI"] if have_ffpi else ["BFI-2"]; return np.mean([base[t][c][0] for t in ts])
def pscore(pid):
    insts={}
    for k,pth in [("BFI-2",f"results_worldtour/{pid}.json"),("FFPI",f"results_worldtour_ffpi/{pid}.json")]:
        if os.path.exists(pth): insts[k]=json.load(open(pth))["table"][k]
    return {c:np.mean([insts[k][c][0] for k in insts]) for c in CODES} if insts else None
rows=[]
for pid,m in meta.items():
    s=pscore(pid)
    if s: rows.append(dict(country=m["country"],sex=m["sex"],ab=m["ab"],**s))
df=pd.DataFrame(rows)
print(f"loaded {len(df)} | {'BFI-2+FFPI' if have_ffpi else 'BFI-2'}")

# x positions: country-major, age-minor
groups=[(co,ab) for co in COUNTRIES for ab in ABS]
def mci(v): v=np.asarray(v); return v.mean(),1.96*v.std(ddof=1)/np.sqrt(len(v))
fig,axes=plt.subplots(5,1,figsize=(34,26))
for ax,c,name in zip(axes,CODES,NAMES):
    allv=[]; vios=[]  # (body, mean, col) for every violin in this row
    for xi,(co,ab) in enumerate(groups):
        sub=df[(df.country==co)&(df.ab==ab)]
        M=sub[sub.sex=="male"][c].values; F=sub[sub.sex=="female"][c].values
        for vals,off,col in [(M,-0.18,BLUE),(F,0.18,RED)]:
            if len(vals)<2: continue
            vp=ax.violinplot(vals,[xi+off],widths=0.32,showextrema=False)
            body=vp['bodies'][0]; body.set_facecolor(col); body.set_alpha(.45)
            vios.append((body,float(np.mean(vals)),col))
            allv+=list(vals)
        if len(M)>=2 and len(F)>=2:
            mM,cM=mci(M); mF,cF=mci(F); _,p=sps.ttest_ind(M,F)
            ax.errorbar(xi-0.18,mM,yerr=cM,fmt='o',color=BLUE,ms=5,capsize=3,mec='black',zorder=6)
            ax.errorbar(xi+0.18,mF,yerr=cF,fmt='o',color=RED,ms=5,capsize=3,mec='black',zorder=6)
    # highlight the two violins with the largest gap of means (max-mean & min-mean), darker in their own colour
    if vios:
        hi=max(vios,key=lambda t:t[1]); lo=min(vios,key=lambda t:t[1])
        for body,mn,col in (hi,lo):
            body.set_facecolor("#6e0000" if col==RED else "#0d1f3c"); body.set_alpha(.95)
    b=base_score(c); ax.axhline(b,ls="--",color="black",lw=1)
    if allv: ax.set_ylim(min(min(allv),b)-0.05,max(max(allv),b)+0.08)
    # country separators + labels
    for k in range(1,len(COUNTRIES)): ax.axvline(k*len(ABS)-0.5,color="#cccccc",lw=1)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([ab for (_,ab) in groups],rotation=45)
    ax.set_ylabel(f"{name} (1-5)")
plt.tight_layout(); plt.savefig(f"{OUT}/worldtour_sex_by_country_and_age.png",dpi=120); plt.close()
print("wrote worldtour_sex_by_country_and_age.png ->",OUT)
