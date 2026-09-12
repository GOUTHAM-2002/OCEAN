"""Phase-B user plots for WizardLM-33B-Uncensored: per user x beta, 3 bars
(Baseline + BFI-2 + FFPI) with the dashed DPO'd-user actual-OCEAN reference line.
Mirrors replot20.py make_plot. Output g_plots/WizardLM/<SHORT>/user_<uid>/."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "WizardLM-33B-Uncensored"
OUTD = os.path.join(HERE, "g_plots", "WizardLM", SHORT)
CODES = ["O", "C", "E", "A", "N"]
NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TESTS = ["BFI-2", "FFPI"]; SERIES = ["Baseline (SFT)"] + TESTS
COLORS = ["#7f7f7f", "#4C72B0", "#C44E52"]; BETAS = ["0.01", "0.1", "0.5"]
USERS = json.load(open(os.path.join(HERE, "user_ids.json")))

resp = pd.read_csv(os.path.join(HERE, "ipip300_responses_and_scores.csv")); resp["case_id"] = resp["case_id"].astype(str)
SC = {"O": "Openness_score", "C": "Conscientiousness_score", "E": "Extraversion_score",
      "A": "Agreeableness_score", "N": "Neuroticism_score"}
USER_SCORES = {int(u): {c: float(resp.loc[resp.case_id == str(u), col].iloc[0]) / 60 for c, col in SC.items()}
               for u in USERS}

def load(label):
    p = os.path.join(HERE, f"results20_{SHORT}_{label}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None

base = load("baseline")

def make_plot(dpo_tbl, outfile, user_scores):
    b = {c: [np.mean([base[t][c][0] for t in TESTS]), np.mean([base[t][c][1] for t in TESTS])] for c in CODES}
    x = np.arange(5); w = 0.26; fig, ax = plt.subplots(figsize=(10, 5.5))
    for si, s in enumerate(SERIES):
        ms = []; cs = []
        for c in CODES:
            m, ci = b[c] if s == "Baseline (SFT)" else dpo_tbl[s][c]; ms.append(m); cs.append(ci)
        ax.bar(x + (si - 1) * w, ms, w, yerr=cs, capsize=4, label=s, color=COLORS[si],
               edgecolor="black", linewidth=0.5)
    for i, c in enumerate(CODES):
        ax.plot([i - 0.42, i + 0.42], [user_scores[c]] * 2, ls="--", color="black", lw=1.8,
                label=("DPO'd user actual" if i == 0 else None))
    ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("Score (1-5)")
    ax.set_ylim(1, 5); ax.legend()
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

n = 0
for uid in USERS:
    od = os.path.join(OUTD, f"user_{uid}"); os.makedirs(od, exist_ok=True)
    for b in BETAS:
        t = load(f"user_{uid}_beta{b}")
        if t:
            make_plot(t, os.path.join(od, f"user_{uid}_beta{b}.png"), USER_SCORES[uid]); n += 1
print(f"Phase-B user plots written: {n} -> {OUTD}")
