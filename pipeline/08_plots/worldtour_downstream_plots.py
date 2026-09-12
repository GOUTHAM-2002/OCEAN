"""One violin figure per downstream task, in the loved country x age x sex style.
Each violin = the (up to 25) people in that country/age/sex cell; y = that adapter's
task score. Baseline model = dashed line. Max-gap pair of violins highlighted darker.
Works on partial results. Figures are written to ``the_last_dance`` and, together
with the extracted per-adapter/grouped data, to the WizardLM-13B downstream folder.
"""
import json, os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sps

# Keep the established figure design unchanged, but make every text element
# legible after the plots are reduced for a paper.
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
})

OUT = "g_plots/the_last_dance"; os.makedirs(OUT, exist_ok=True)
DOWNSTREAM_OUT = "g_plots/WizardLM/WizardLM-13B-V1.2/downstream"
os.makedirs(DOWNSTREAM_OUT, exist_ok=True)
RES = "results_downstream_worldtour"
BLUE="#4C72B0"; RED="#C44E52"; DBLUE="#0d1f3c"; DRED="#6e0000"
COUNTRIES=["USA","India","China","South Afric","Germany","Australia","Brazil"]; CLAB={"South Afric":"S.Afr"}
ABS=["18-24","25-34","35+"]
meta={int(k):v for k,v in json.load(open("worldtour_meta.json")).items()}

# (display, result_key, metric) ; sycophancy is special (mean of 3 subtasks)
TASKS=[("HHH alignment","hhh_alignment","acc,none"),
       ("Sycophancy (mean of 3)","__syco__",None),
       ("ETHICS commonsense","ethics_cm","acc,none"),
       ("ETHICS deontology","ethics_deontology","acc,none"),
       ("ETHICS justice","ethics_justice","acc,none"),
       ("ETHICS utilitarianism","ethics_utilitarianism","acc,none"),
       ("ETHICS virtue","ethics_virtue","acc,none"),
       ("Moral stories (acc_norm)","moral_stories","acc_norm,none"),
       ("CrowS-Pairs (%stereotype, lower=better)","crows_pairs_english","pct_stereotype,none")]

def load_results(label):
    for p in glob.glob(os.path.join(RES,label,"**","results_*.json"),recursive=True):
        try: return json.load(open(p))["results"]
        except Exception: pass
    return None
def task_val(r,key,metric):
    if r is None: return None
    if key=="__syco__":
        vs=[r[k].get("acc,none") for k in ("sycophancy_on_nlp_survey","sycophancy_on_philpapers2020","sycophancy_on_political_typology_quiz") if k in r]
        return float(np.mean(vs)) if len(vs)==3 else None
    return r[key].get(metric) if key in r else None

base_r=load_results("baseline")
rows=[]
for pid,m in meta.items():
    r=load_results(str(pid))
    if r is None: continue
    rec=dict(adapter_id=pid,country=m["country"],sex=m["sex"],age_bracket=m["ab"])
    for disp,key,met in TASKS: rec[disp]=task_val(r,key,met)
    rows.append(rec)
df=pd.DataFrame(rows)
print(f"loaded {len(df)} adapters with downstream results")
if len(df)==0: raise SystemExit("no results yet")

# Persist the exact values used by the figures so downstream analyses do not
# need to re-extract lm-eval's nested result JSONs.
raw_path=f"{DOWNSTREAM_OUT}/worldtour_downstream_results.csv"
df.sort_values("adapter_id").to_csv(raw_path,index=False,float_format="%.8g")
baseline_rows=[]
for disp,key,met in TASKS:
    baseline_rows.append({"task":disp,"baseline_score":task_val(base_r,key,met)})
pd.DataFrame(baseline_rows).to_csv(
    f"{DOWNSTREAM_OUT}/worldtour_downstream_baseline.csv",index=False,float_format="%.8g")

long_df=df.melt(id_vars=["adapter_id","country","sex","age_bracket"],
                var_name="task",value_name="score").dropna(subset=["score"])
grouped=(long_df.groupby(["task","country","age_bracket","sex"],sort=True)["score"]
         .agg(n="count",mean="mean",std="std").reset_index())
grouped["sem"]=grouped["std"]/np.sqrt(grouped["n"])
grouped["ci95"]=1.96*grouped["sem"]
grouped.to_csv(f"{DOWNSTREAM_OUT}/worldtour_downstream_group_summary.csv",
               index=False,float_format="%.8g")
print("  wrote",raw_path)

groups=[(co,ab) for co in COUNTRIES for ab in ABS]
def mci(v): v=np.asarray(v,float); v=v[~np.isnan(v)]; return (v.mean(),1.96*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else (np.nan,0)

for disp,key,met in TASKS:
    col=disp
    if df[col].notna().sum()<20:
        print(f"  skip {disp}: only {df[col].notna().sum()} values"); continue
    fig,ax=plt.subplots(figsize=(30,6)); vios=[]
    for xi,(co,ab) in enumerate(groups):
        sub=df[(df.country==co)&(df.age_bracket==ab)]
        M=sub[sub.sex=="male"][col].dropna().values; F=sub[sub.sex=="female"][col].dropna().values
        for vals,off,c in [(M,-0.18,BLUE),(F,0.18,RED)]:
            if len(vals)<2: continue
            vp=ax.violinplot(vals,[xi+off],widths=0.32,showextrema=False)
            b=vp['bodies'][0]; b.set_facecolor(c); b.set_alpha(.45); vios.append((b,float(np.mean(vals)),c))
        if len(M)>=2 and len(F)>=2:
            mM,cM=mci(M); mF,cF=mci(F); _,p=sps.ttest_ind(M,F,nan_policy='omit')
            ax.errorbar(xi-0.18,mM,yerr=cM,fmt='o',color=BLUE,ms=5,capsize=3,mec='black',zorder=6)
            ax.errorbar(xi+0.18,mF,yerr=cF,fmt='o',color=RED,ms=5,capsize=3,mec='black',zorder=6)
            if p<0.05: ax.text(xi,max(mM,mF)+0.01,"*",ha='center',fontsize=13,color='green')
    if vios:  # highlight the max-mean and min-mean violins (largest gap)
        hi=max(vios,key=lambda t:t[1]); lo=min(vios,key=lambda t:t[1])
        for b,mn,c in (hi,lo): b.set_facecolor(DRED if c==RED else DBLUE); b.set_alpha(.95)
    bval=task_val(base_r,key,met)
    if bval is not None: ax.axhline(bval,ls="--",color="black",lw=1.2,label=f"baseline={bval:.3f}")
    for k in range(1,len(COUNTRIES)): ax.axvline(k*3-0.5,color="#cccccc",lw=1)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([ab for _,ab in groups],fontsize=8,rotation=45)
    y1=ax.get_ylim()[1]
    for ci,co in enumerate(COUNTRIES): ax.text(ci*3+1-0.5,y1,CLAB.get(co,co),ha='center',va='bottom',fontsize=11,fontweight='bold')
    ax.set_ylabel(disp,fontsize=12); ax.legend(loc='lower right',fontsize=9)
    fn=f"{OUT}/downstream_{key if key!='__syco__' else 'sycophancy'}.png"
    plt.tight_layout(); plt.savefig(fn,dpi=120)
    downstream_fn=(f"{DOWNSTREAM_OUT}/worldtour_downstream_"
                   f"{key if key!='__syco__' else 'sycophancy'}.png")
    plt.savefig(downstream_fn,dpi=120)
    plt.close(); print("  wrote",fn,"and",downstream_fn)
print("done")
