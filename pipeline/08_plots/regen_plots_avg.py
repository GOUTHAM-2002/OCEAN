"""
Regenerate per-config plots for the single-trait DPO sweep from the existing
result JSONs (no recompute). For every trait config of every model:
  Plot A (per test):  baseline-avg + DPO on each of BFI-2/Goldberg-100/FFPI
                      -> g_plots/<family>/<short>/TRAIT_<T>_<DIR>/trait_<T>_<dir>_beta<b>.png
  Plot B (averaged):  baseline-avg + DPO averaged over BFI-2 + FFPI (Goldberg-100 dropped)
                      -> g_plots/<family>/<short>/TRAIT_<T>_<DIR>/trait_<T>_<dir>_beta<b>_avg.png
Reads only results_<short>_*.json that already exist; writes only plot PNGs.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__))
MODELS=[
 ("Tulu-2","tulu-2-7b"),("Tulu-2","tulu-2-13b"),
 ("Vicuna","vicuna-7b-v1.5"),("Vicuna","vicuna-13b-v1.5"),
 ("WizardLM","wizardLM-7B"),("WizardLM","WizardLM-13B-V1.2"),
]
BETAS=["0.01","0.1","0.5"]; TRAITS=["O","C","E","A","N"]; DIRS=["high","low"]
codes=["O","C","E","A","N"]
names=["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
gen_tests=["BFI-2","Goldberg-100","FFPI"]; series=["Baseline (SFT)"]+gen_tests
avg_tests=["BFI-2","FFPI"]   # tests averaged in the _avg plots (Goldberg-100 dropped)
colors=["#7f7f7f","#4C72B0","#55A868","#C44E52"]

def base_means(base_tbl):
    return {c:[np.mean([base_tbl[t][c][0] for t in gen_tests]),
               np.mean([base_tbl[t][c][1] for t in gen_tests])] for c in codes}

def make_plot(dpo_tbl, base_tbl, outfile, title=""):
    base=base_means(base_tbl)
    x=np.arange(len(codes)); w=0.2; fig,ax=plt.subplots(figsize=(10,5.5))
    for si,s in enumerate(series):
        ms=[]; cs=[]
        for c in codes:
            m,ci=base[c] if s=="Baseline (SFT)" else dpo_tbl[s][c]
            ms.append(m); cs.append(ci)
        ax.bar(x+(si-1.5)*w, ms, w, yerr=cs, capsize=4, label=s, color=colors[si],
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Score (1-5)")
    ax.set_ylim(1,5); ax.legend()
    if title: ax.set_title(title)
    plt.tight_layout(); plt.savefig(outfile,dpi=150); plt.close()

def make_plot_avg(dpo_tbl, base_tbl, outfile, title=""):
    """Baseline average vs DPO average over BFI-2 + FFPI (Goldberg-100 dropped), per OCEAN trait."""
    x=np.arange(len(codes)); w=0.35; fig,ax=plt.subplots(figsize=(9,5.5))
    base_m=[np.mean([base_tbl[t][c][0] for t in avg_tests]) for c in codes]
    base_ci=[np.mean([base_tbl[t][c][1] for t in avg_tests]) for c in codes]
    dpo_m=[np.mean([dpo_tbl[t][c][0] for t in avg_tests]) for c in codes]
    dpo_ci=[np.mean([dpo_tbl[t][c][1] for t in avg_tests]) for c in codes]
    ax.bar(x-w/2, base_m, w, yerr=base_ci, capsize=4, label="Baseline (SFT) avg",
           color="#7f7f7f", edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, dpo_m, w, yerr=dpo_ci, capsize=4, label="DPO avg (BFI-2 + FFPI)",
           color="#C44E52", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Score (1-5)")
    ax.set_ylim(1,5); ax.legend()
    if title: ax.set_title(title)
    plt.tight_layout(); plt.savefig(outfile,dpi=150); plt.close()

nA=nB=0
for family, short in MODELS:
    bf=f"results_{short}_baseline.json"
    if not os.path.exists(bf):
        print(f"SKIP {short}: no baseline"); continue
    base=json.load(open(bf))["table"]
    for trait in TRAITS:
        for d in DIRS:
            od=os.path.join(HERE,"g_plots",family,short,f"TRAIT_{trait}_{d.upper()}")
            os.makedirs(od,exist_ok=True)
            for beta in BETAS:
                rf=f"results_{short}_trait{trait}_{d}_beta{beta}.json"
                if not os.path.exists(rf): continue
                tbl=json.load(open(rf))["table"]
                ttl=f"{short}  {trait}-{d}  beta={beta}"
                make_plot(tbl, base, os.path.join(od,f"trait_{trait}_{d}_beta{beta}.png"), title=ttl); nA+=1
                make_plot_avg(tbl, base, os.path.join(od,f"trait_{trait}_{d}_beta{beta}_avg.png"),
                              title=ttl+"  (avg over BFI-2 + FFPI)"); nB+=1
    print(f"  {short}: done")
print(f"REGEN_DONE  plotA={nA}  plotB={nB}")
