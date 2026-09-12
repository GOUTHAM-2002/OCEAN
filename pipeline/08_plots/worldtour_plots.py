"""World Tour — Male vs Female DPO across 7 countries. Individual-score violins
(loved style) with BOLD mean±95%CI markers, the M-F gap annotated, tight y-zoom
so the difference is visible, baseline reference, significance. Uses combined
BFI-2 + FFPI when FFPI results exist, else BFI-2 only. -> g_plots/the_last_dance/."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sps

OUT = "g_plots/the_last_dance"; os.makedirs(OUT, exist_ok=True)
CODES = ["O","C","E","A","N"]; NAMES = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
BLUE="#4C72B0"; RED="#C44E52"
COUNTRIES = ["USA","India","China","South Afric","Germany","Australia","Brazil"]
CLABEL = {"South Afric":"S.Africa"}
meta = {int(k):v for k,v in json.load(open("worldtour_meta.json")).items()}
base = json.load(open("results20_WizardLM-13B-V1.2_baseline.json"))["table"]

def person_score(pid):
    insts = {}
    p1=f"results_worldtour/{pid}.json"; p2=f"results_worldtour_ffpi/{pid}.json"
    if os.path.exists(p1): insts["BFI-2"]=json.load(open(p1))["table"]["BFI-2"]
    if os.path.exists(p2): insts["FFPI"]=json.load(open(p2))["table"]["FFPI"]
    if not insts: return None
    return {c: np.mean([insts[k][c][0] for k in insts]) for c in CODES}
have_ffpi = os.path.isdir("results_worldtour_ffpi") and len(os.listdir("results_worldtour_ffpi"))>500
def base_score(c):
    ts = ["BFI-2","FFPI"] if have_ffpi else ["BFI-2"]
    return np.mean([base[t][c][0] for t in ts])

rows=[]
for pid,m in meta.items():
    s=person_score(pid)
    if s is None: continue
    rows.append(dict(country=m["country"],sex=m["sex"],ab=m["ab"],**s))
df=pd.DataFrame(rows)
print(f"loaded {len(df)} people | instruments: {'BFI-2+FFPI' if have_ffpi else 'BFI-2 only'}")

def mci(v): v=np.asarray(v); return v.mean(), 1.96*v.std(ddof=1)/np.sqrt(len(v))
def draw(ax,xc,M,F,w=0.32,annotate=True):
    for vals,off,col in [(M,-0.2,BLUE),(F,0.2,RED)]:
        if len(vals)<2: continue
        vp=ax.violinplot(vals,[xc+off],widths=w,showextrema=False)
        for b in vp['bodies']: b.set_facecolor(col); b.set_alpha(.45)
    mM,cM=mci(M); mF,cF=mci(F)
    ax.errorbar(xc-0.2,mM,yerr=cM,fmt='o',color=BLUE,ms=8,capsize=5,mec='black',mew=1.2,zorder=6)
    ax.errorbar(xc+0.2,mF,yerr=cF,fmt='o',color=RED,ms=8,capsize=5,mec='black',mew=1.2,zorder=6)
    if annotate:
        _,p=sps.ttest_ind(M,F); d=mF-mM
        ax.annotate("", xy=(xc+0.2,mF),xytext=(xc-0.2,mM),arrowprops=dict(arrowstyle='-',lw=1,color='k',alpha=.5))
        ax.text(xc, max(mM,mF)+max(cM,cF)+0.01, f"Δ{d:+.2f}{'*' if p<0.05 else ''}",
                ha='center',fontsize=8,fontweight='bold' if p<0.05 else 'normal')

fig, axes = plt.subplots(5,1,figsize=(15,24))
for ax,c,name in zip(axes,CODES,NAMES):
    allv=[]
    for xi,co in enumerate(COUNTRIES):
        M=df[(df.country==co)&(df.sex=="male")][c].values; F=df[(df.country==co)&(df.sex=="female")][c].values
        draw(ax,xi,M,F); allv+=list(M)+list(F)
    b=base_score(c); ax.axhline(b,ls="--",color="black",lw=1.3,label="baseline")
    lo=min(min(allv),b)-0.05; hi=max(max(allv),b)+0.08
    ax.set_ylim(lo,hi)  # tight zoom so the M-F gap is visible
    ax.set_xticks(range(len(COUNTRIES))); ax.set_xticklabels([CLABEL.get(x,x) for x in COUNTRIES])
    ax.set_ylabel(f"{name} (1-5)")
axes[0].set_title(f"DPO on Male (blue) vs Female (red) across 7 countries — individual people; big dot=mean±95%CI; Δ=female−male (*p<.05); dashed=baseline  [{'BFI-2+FFPI' if have_ffpi else 'BFI-2'}]",fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/worldtour_sex_by_country.png",dpi=140); plt.close()
print("wrote worldtour_sex_by_country.png")
