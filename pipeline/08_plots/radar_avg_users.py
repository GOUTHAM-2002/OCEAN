"""
Per-model radar chart of the OCEAN profile AVERAGED over all 20 DPO'd users.

For each model with per-user results (results20_<short>_user_<uid>_beta<b>.json):
  - Average each DPO'd user's OCEAN score over BFI-2 + FFPI (Goldberg dropped,
    same as replot20.py), then average across all users -> one line per beta.
  - Baseline (SFT) line: baseline table averaged over BFI-2 + FFPI.
  - Target line: users' actual OCEAN (1-5), averaged across users.

Writes:  g_plots/<family>/<short>/avg_users_radar.png
"""
import os, json
import numpy as np, pandas as pd
if not hasattr(np, "float"):       # old matplotlib polar code references the removed np.float alias
    np.float = float
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = ["O", "C", "E", "A", "N"]
NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TESTS = ["BFI-2", "FFPI"]                      # Goldberg dropped (matches replot20)
BETAS = ["0.01", "0.1", "0.5"]
SMALL = ["tulu-2-7b", "tulu-2-13b", "vicuna-7b-v1.5", "vicuna-13b-v1.5", "wizardLM-7B", "WizardLM-13B-V1.2"]
FAMILY = {"tulu-2-7b": "Tulu-2", "tulu-2-13b": "Tulu-2", "vicuna-7b-v1.5": "Vicuna",
          "vicuna-13b-v1.5": "Vicuna", "wizardLM-7B": "WizardLM", "WizardLM-13B-V1.2": "WizardLM"}
BETA_COLORS = {"0.01": "#4C72B0", "0.1": "#55A868", "0.5": "#C44E52"}
USERS = json.load(open(os.path.join(HERE, "user_ids.json")))

# users' actual OCEAN (1-5), scaled from the 60-point IPIP-300 trait scores
resp = pd.read_csv(os.path.join(HERE, "ipip300_responses_and_scores.csv")); resp["case_id"] = resp["case_id"].astype(str)
SC = {"O": "Openness_score", "C": "Conscientiousness_score", "E": "Extraversion_score",
      "A": "Agreeableness_score", "N": "Neuroticism_score"}
USER_SCORES = {int(u): {c: float(resp.loc[resp.case_id == str(u), col].iloc[0]) / 60 for c, col in SC.items()}
               for u in USERS}

def load(short, label):
    p = os.path.join(HERE, f"results20_{short}_{label}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None

def avg_over_tests(tbl):
    """OCEAN mean over BFI-2 + FFPI for one result table -> dict code->mean."""
    return {c: float(np.mean([tbl[t][c][0] for t in TESTS])) for c in CODES}

def close(vals):
    """Repeat the first value at the end so the radar polygon closes."""
    return list(vals) + [vals[0]]

def make_radar(short, outfile):
    base_tbl = load(short, "baseline")
    if not base_tbl:
        return False
    baseline = avg_over_tests(base_tbl)

    # target: mean actual OCEAN across all users
    target = {c: float(np.mean([USER_SCORES[int(u)][c] for u in USERS])) for c in CODES}

    # DPO: average each user's per-test mean, then across users, per beta
    dpo = {}
    for b in BETAS:
        per_user = []
        for u in USERS:
            t = load(short, f"user_{u}_beta{b}")
            if t:
                per_user.append(avg_over_tests(t))
        if per_user:
            dpo[b] = {c: float(np.mean([pu[c] for pu in per_user])) for c in CODES}

    angles = np.linspace(0, 2 * np.pi, len(CODES), endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)

    # filled (not hollow). DPO betas first / underneath, then baseline, then the
    # user target as a bold line on top so the user-vs-DPO gap is easy to read.
    for b in BETAS:
        if b in dpo:
            vals = close([dpo[b][c] for c in CODES])
            ax.plot(angles_c, vals, color=BETA_COLORS[b], lw=2, marker="o", ms=4,
                    label=f"DPO avg (beta={b})")
            ax.fill(angles_c, vals, color=BETA_COLORS[b], alpha=0.15)
    base_vals = close([baseline[c] for c in CODES])
    ax.plot(angles_c, base_vals, color="#7f7f7f", lw=2, marker="o", ms=4, label="Baseline (SFT)")
    ax.fill(angles_c, base_vals, color="#7f7f7f", alpha=0.12)
    tgt_vals = close([target[c] for c in CODES])
    ax.fill(angles_c, tgt_vals, color="black", alpha=0.06)
    ax.plot(angles_c, tgt_vals, ls="--", color="black", lw=3, marker="D", ms=6,
            zorder=10, label="User actual (avg)")

    # zoom the radial axis to the data so small user-vs-DPO differences are visible
    allv = [v for c in CODES for v in
            ([target[c], baseline[c]] + [dpo[b][c] for b in BETAS if b in dpo])]
    lo = max(1.0, np.floor((min(allv) - 0.05) * 20) / 20)
    hi = min(5.0, np.ceil((max(allv) + 0.05) * 20) / 20)
    ax.set_xticks(angles); ax.set_xticklabels(NAMES)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.round(np.linspace(lo, hi, 5), 1))
    ax.tick_params(axis="y", labelsize=8); ax.set_rlabel_position(90)
    ax.set_title(f"{short}  -  OCEAN averaged over all {len(USERS)} users\n(radial axis zoomed to data)", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9)
    plt.tight_layout(); plt.savefig(outfile, dpi=150, bbox_inches="tight"); plt.close()
    return True

n = 0
for short in SMALL:
    od = os.path.join(HERE, "g_plots", FAMILY[short], short); os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "avg_users_radar.png")
    if make_radar(short, out):
        print(f"  wrote {out}"); n += 1
    else:
        print(f"  SKIP {short}: no baseline")
print(f"RADAR_DONE  {n} model radar charts")
