"""
Regenerate ALL personality plots from the 20-seed results (results20_*.json),
dropping Goldberg-100. 3 bars/trait: Baseline (mean of BFI-2+FFPI) + BFI-2 + FFPI.
Overwrites the existing PNGs in g_plots/. User plots keep the dashed actual-score line.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = ["O", "C", "E", "A", "N"]
NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TESTS = ["BFI-2", "FFPI"]                      # Goldberg dropped
SERIES = ["Baseline (SFT)"] + TESTS
COLORS = ["#7f7f7f", "#4C72B0", "#C44E52"]
BETAS = ["0.01", "0.1", "0.5"]
SMALL = ["tulu-2-7b", "tulu-2-13b", "vicuna-7b-v1.5", "vicuna-13b-v1.5", "wizardLM-7B", "WizardLM-13B-V1.2"]
FAMILY = {"tulu-2-7b": "Tulu-2", "tulu-2-13b": "Tulu-2", "vicuna-7b-v1.5": "Vicuna",
          "vicuna-13b-v1.5": "Vicuna", "wizardLM-7B": "WizardLM", "WizardLM-13B-V1.2": "WizardLM"}
USERS = json.load(open(os.path.join(HERE, "user_ids.json")))

# user actual OCEAN (1-5) for dashed lines
resp = pd.read_csv(os.path.join(HERE, "ipip300_responses_and_scores.csv")); resp["case_id"] = resp["case_id"].astype(str)
SC = {"O": "Openness_score", "C": "Conscientiousness_score", "E": "Extraversion_score",
      "A": "Agreeableness_score", "N": "Neuroticism_score"}
USER_SCORES = {int(u): {c: float(resp.loc[resp.case_id == str(u), col].iloc[0]) / 60 for c, col in SC.items()}
               for u in USERS}

def load(short, label):
    p = os.path.join(HERE, f"results20_{short}_{label}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None

def make_plot(dpo_tbl, base_tbl, outfile, user_scores=None):
    base = {c: [np.mean([base_tbl[t][c][0] for t in TESTS]),
                np.mean([base_tbl[t][c][1] for t in TESTS])] for c in CODES}
    x = np.arange(5); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for si, s in enumerate(SERIES):
        ms = []; cs = []
        for c in CODES:
            m, ci = base[c] if s == "Baseline (SFT)" else dpo_tbl[s][c]
            ms.append(m); cs.append(ci)
        ax.bar(x + (si - 1) * w, ms, w, yerr=cs, capsize=4, label=s, color=COLORS[si],
               edgecolor="black", linewidth=0.5)
    if user_scores:
        for i, c in enumerate(CODES):
            ax.plot([i - 0.42, i + 0.42], [user_scores[c]] * 2, ls="--", color="black", lw=1.8,
                    label=("DPO'd user actual" if i == 0 else None))
    ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("Score (1-5)")
    ax.set_ylim(1, 5); ax.legend()
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

n = 0
# ---- Mixtral ----
mb = load("mixtral", "baseline")
if mb:
    for a, folder in [("low", "AGENT_LOW"), ("high", "AGENT_HIGH")]:
        od = os.path.join(HERE, "g_plots", "Nous-Hermes-2-Mixtral-8x7B-SFT", folder); os.makedirs(od, exist_ok=True)
        for b in BETAS:
            t = load("mixtral", f"{a}_beta{b}")
            if t: make_plot(t, mb, os.path.join(od, f"{folder.lower()}_beta{b}.png")); n += 1
# ---- small models ----
for short in SMALL:
    bt = load(short, "baseline")
    if not bt: continue
    fam = FAMILY[short]
    for a, folder in [("low", "AGENT_LOW"), ("high", "AGENT_HIGH")]:
        od = os.path.join(HERE, "g_plots", fam, short, folder); os.makedirs(od, exist_ok=True)
        for b in BETAS:
            t = load(short, f"{a}_beta{b}")
            if t: make_plot(t, bt, os.path.join(od, f"{folder.lower()}_beta{b}.png")); n += 1
    for uid in USERS:
        od = os.path.join(HERE, "g_plots", fam, short, f"user_{uid}"); os.makedirs(od, exist_ok=True)
        for b in BETAS:
            t = load(short, f"user_{uid}_beta{b}")
            if t: make_plot(t, bt, os.path.join(od, f"user_{uid}_beta{b}.png"), user_scores=USER_SCORES[uid]); n += 1
print(f"regenerated {n} plots (n=20 seeds, Goldberg dropped, 3 bars)")
