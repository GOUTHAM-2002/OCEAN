"""
Per-user OCEAN bar plots for the vicuna-7b-v1.5 100-user DPO sweep.
Plot code reused from master_users.py (step 3), plotting only -- no training.
Reads results_vicuna-7b-v1.5_user_<uid>_beta<b>.json and writes
  g_plots/Vicuna/vicuna-7b-v1.5/user_<uid>/user_<uid>_beta<b>.png
Skips a plot whose png already exists unless REDO=1. Run after the sweep:
  python3 plots_users_vicuna7b.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY, SHORT = "Vicuna", "vicuna-7b-v1.5"
BETAS = ["0.01", "0.1", "0.5"]
USERS = json.load(open(os.path.join(HERE, "user_ids_100.json")))
REDO = os.environ.get("REDO") == "1"

codes = ["O", "C", "E", "A", "N"]
names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
gen_tests = ["BFI-2", "FFPI"]
series = ["Baseline (SFT)"] + gen_tests
colors = ["#7f7f7f", "#4C72B0", "#C44E52"]   # Baseline grey, BFI-2 blue, FFPI red

# each DPO'd user's ACTUAL OCEAN scores on the 1-5 scale (sums /60), for dashed ref lines
_resp = pd.read_csv(os.path.join(HERE, "ipip300_responses_and_scores.csv"))
_resp["case_id"] = _resp["case_id"].astype(str)
_SC = {"O": "Openness_score", "C": "Conscientiousness_score", "E": "Extraversion_score",
       "A": "Agreeableness_score", "N": "Neuroticism_score"}
USER_SCORES = {int(uid): {c: float(_resp.loc[_resp.case_id == str(uid), col].iloc[0]) / 60.0
                          for c, col in _SC.items()} for uid in USERS}


def make_plot(dpo_tbl, base_tbl, outfile, user_scores=None):
    base = {c: [np.mean([base_tbl[t][c][0] for t in gen_tests]),
                np.mean([base_tbl[t][c][1] for t in gen_tests])] for c in codes}
    x = np.arange(len(codes)); w = 0.2; fig, ax = plt.subplots(figsize=(10, 5.5))
    for si, s in enumerate(series):
        ms = []; cs = []
        for c in codes:
            m, ci = base[c] if s == "Baseline (SFT)" else dpo_tbl[s][c]
            ms.append(m); cs.append(ci)
        ax.bar(x + (si - (len(series) - 1) / 2) * w, ms, w, yerr=cs, capsize=4, label=s,
               color=colors[si], edgecolor="black", linewidth=0.5)
    if user_scores:   # dashed line per component = the DPO'd user's actual OCEAN score
        for i, c in enumerate(codes):
            ax.plot([i - 0.42, i + 0.42], [user_scores[c]] * 2, ls="--", color="black",
                    lw=1.8, label=("DPO'd user actual" if i == 0 else None))
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Score (1-5)")
    ax.set_ylim(1, 5); ax.legend(); plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()


base = json.load(open(os.path.join(HERE, f"results_{SHORT}_baseline.json")))["table"]
n_plotted = n_missing = 0
for uid in USERS:
    od = os.path.join(HERE, "g_plots", FAMILY, SHORT, f"user_{uid}")
    os.makedirs(od, exist_ok=True)
    for beta in BETAS:
        res = os.path.join(HERE, f"results_{SHORT}_user_{uid}_beta{beta}.json")
        png = os.path.join(od, f"user_{uid}_beta{beta}.png")
        if not os.path.exists(res):
            n_missing += 1; continue
        if os.path.exists(png) and not REDO:
            continue
        make_plot(json.load(open(res))["table"], base, png, user_scores=USER_SCORES[uid])
        n_plotted += 1
print(f"PLOTS_DONE plotted={n_plotted} missing_results={n_missing} "
      f"-> g_plots/{FAMILY}/{SHORT}/user_*/", flush=True)
