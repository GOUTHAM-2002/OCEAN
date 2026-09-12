"""Matched 1000-user IPIP vs post-DPO OCEAN trait-correlation heatmaps."""
import csv, glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT="g_plots/correlation"; os.makedirs(OUT,exist_ok=True)
CODES=["O","C","E","A","N"]
NAMES=["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]
COLS=dict(zip(CODES,["Openness_score","Conscientiousness_score","Extraversion_score","Agreeableness_score","Neuroticism_score"]))
MODELS={
 "WizardLM-13B-v1.2":"results_WizardLM-13B-V1.2_user_{uid}_beta0.01.json",
 "Tulu-2-7B":"results_tulu-2-7b_user_{uid}_beta0.01.json",
 "Vicuna-13B-v1.5":"results_vicuna-13b-v1.5_user_{uid}_beta0.01.json",
 "Nous-Hermes Mixtral":"results_Nous-Hermes-2-Mixtral-8x7B-SFT_user_{uid}_beta0.01.json",
}
USERS=sorted(json.load(open("user_ids_1000.json")))
human={}
with open("ipip300_responses_and_scores_10000.csv") as f:
    for row in csv.DictReader(f):
        try: uid=int(row["case_id"])
        except (ValueError,TypeError): continue
        if uid in USERS: human[uid]=[float(row[COLS[c]])/60 for c in CODES]

def model_row(path):
    t=json.load(open(path))["table"]
    return [float(np.mean([t[q][c][0] for q in ("BFI-2","FFPI")])) for c in CODES]

valid=[]
for uid in USERS:
    if uid not in human: continue
    if all(os.path.exists(p.format(uid=uid)) for p in MODELS.values()): valid.append(uid)
assert len(valid)==1000, f"expected matched n=1000, got {len(valid)}"
data={"IPIP-NEO-300 users":np.array([human[u] for u in valid])}
for name,p in MODELS.items(): data[name]=np.array([model_row(p.format(uid=u)) for u in valid])
mats={name:np.corrcoef(x,rowvar=False) for name,x in data.items()}
cmap=LinearSegmentedColormap.from_list("peach_brown",["#ffffff","#fee8d6","#f4a261","#c65d21","#5b2416"])

def draw(ax,mat,title,show_y=True):
    im=ax.imshow(np.abs(mat),vmin=0,vmax=1,cmap=cmap)
    ax.set_xticks(range(5)); ax.set_xticklabels(NAMES,rotation=42,ha="right",fontsize=9)
    ax.set_yticks(range(5)); ax.set_yticklabels(NAMES if show_y else [],fontsize=9)
    ax.set_title(title,fontsize=12,fontweight="bold")
    for i in range(5):
        for j in range(5):
            v=mat[i,j]
            ax.text(j,i,f"{v:+.2f}",ha="center",va="center",fontsize=9,
                    color="white" if abs(v)>.72 else "black")
    return im

fig,axs=plt.subplots(1,5,figsize=(24,5.2),constrained_layout=True)
for i,((name,mat),ax) in enumerate(zip(mats.items(),axs)):
    im=draw(ax,mat,name,i==0)
cbar=fig.colorbar(im,ax=axs,shrink=.78,pad=.015)
cbar.set_label("Correlation magnitude $|r|$ (signed $r$ shown in cells)")
fig.suptitle("Matched trait-correlation structure: IPIP-300 users vs post-DPO models\n"
             "$n=1000$ matched users, $\\beta=0.01$; model scores average BFI-2 and FFPI",fontsize=15)
fig.savefig(f"{OUT}/ipip_vs_post_dpo_trait_correlations.png",dpi=220,bbox_inches="tight"); plt.close(fig)

# Polarity-aligned diagnostic: Emotional Stability is the reverse pole of
# Neuroticism. This preserves the raw plot above while making the orientation
# issue explicit rather than silently flipping signs.
aligned_human=mats["IPIP-NEO-300 users"].copy()
aligned_human[4,:]*=-1; aligned_human[:,4]*=-1; aligned_human[4,4]=1
aligned={"IPIP-300 (Emotional Stability = $6-N$)":aligned_human,
         **{k:v for k,v in mats.items() if k!="IPIP-NEO-300 users"}}
fig,axs=plt.subplots(1,5,figsize=(24,5.2),constrained_layout=True)
for i,((name,mat),ax) in enumerate(zip(aligned.items(),axs)):
    im=draw(ax,mat,name,i==0)
cbar=fig.colorbar(im,ax=axs,shrink=.78,pad=.015)
cbar.set_label("Correlation magnitude $|r|$ (signed $r$ shown in cells)")
fig.suptitle("Polarity-aligned trait-correlation structure\n"
             "$n=1000$ matched users; IPIP Neuroticism shown as Emotional Stability ($6-N$)",fontsize=15)
fig.savefig(f"{OUT}/ipip_vs_post_dpo_trait_correlations_polarity_aligned.png",dpi=220,bbox_inches="tight"); plt.close(fig)

for name,mat in mats.items():
    fig,ax=plt.subplots(figsize=(7.4,6.4)); im=draw(ax,mat,f"{name} (n=1000)")
    cb=fig.colorbar(im,ax=ax); cb.set_label("Correlation magnitude $|r|$")
    fig.tight_layout(); safe=re.sub(r"[^a-z0-9]+","_",name.lower()).strip("_")
    fig.savefig(f"{OUT}/{safe}_trait_correlation.png",dpi=220,bbox_inches="tight"); plt.close(fig)

human_vec=mats["IPIP-NEO-300 users"][np.triu_indices(5,1)]
with open(f"{OUT}/correlation_matrices.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["source","trait_1","trait_2","pearson_r"])
    for name,mat in mats.items():
        for i in range(5):
            for j in range(i+1,5): w.writerow([name,NAMES[i],NAMES[j],f"{mat[i,j]:.6f}"])
with open(f"{OUT}/matrix_similarity_to_ipip.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["model","n_users","raw_matrix_similarity","polarity_aligned_similarity","raw_mean_absolute_difference","polarity_aligned_mean_absolute_difference"])
    aligned_vec=aligned_human[np.triu_indices(5,1)]
    for name,mat in list(mats.items())[1:]:
        v=mat[np.triu_indices(5,1)]
        w.writerow([name,len(valid),f"{np.corrcoef(human_vec,v)[0,1]:.6f}",f"{np.corrcoef(aligned_vec,v)[0,1]:.6f}",f"{np.mean(np.abs(human_vec-v)):.6f}",f"{np.mean(np.abs(aligned_vec-v)):.6f}"])
print(f"wrote {len(mats)+2} plots and CSVs to {OUT}; matched n={len(valid)}")
