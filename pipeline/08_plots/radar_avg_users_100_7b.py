"""
Radar chart of the OCEAN profile averaged over the 100 current users, for
WizardLM-13B-V1.2 only. 100-user version of radar_avg_users.py.

Differences from the original 20-user script:
  - user list  : user_ids_100.json (the 100 stratified users)
  - result files: results_<short>_<label>.json  (current naming, not results20_)
  - model       : WizardLM-13B-V1.2 only
  - tests       : BFI-2 + FFPI (Goldberg already dropped from these results)

For each beta, each user's DPO'd model OCEAN is the mean over BFI-2 + FFPI;
those are then averaged across the 100 users -> one line per beta. The baseline
(SFT) line and the users' actual-OCEAN average (target) are drawn for reference.

Writes: g_plots/WizardLM/WizardLM-13B-V1.2/avg_users_radar_100.png
"""
import os, json
import numpy as np, pandas as pd
if not hasattr(np, "float"):       # old matplotlib polar code references the removed np.float alias
    np.float = float
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = ["O", "C", "E", "A", "N"]
NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TESTS = ["BFI-2", "FFPI"]                      # Goldberg dropped from the results
BETAS = ["0.01", "0.1", "0.5"]
FAMILY, SHORT = "WizardLM", "wizardLM-7B"
BETA_COLORS = {"0.01": "#4C72B0", "0.1": "#55A868", "0.5": "#C44E52"}
USERS = json.load(open(os.path.join(HERE, "user_ids_100.json")))

# users' actual OCEAN (1-5), scaled from the 60-point IPIP-300 trait scores
resp = pd.read_csv(os.path.join(HERE, "ipip300_responses_and_scores.csv")); resp["case_id"] = resp["case_id"].astype(str)
SC = {"O": "Openness_score", "C": "Conscientiousness_score", "E": "Extraversion_score",
      "A": "Agreeableness_score", "N": "Neuroticism_score"}
USER_SCORES = {int(u): {c: float(resp.loc[resp.case_id == str(u), col].iloc[0]) / 60 for c, col in SC.items()}
               for u in USERS}

def load(label):
    p = os.path.join(HERE, f"results_{SHORT}_{label}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None

def avg_over_tests(tbl):
    """OCEAN mean over BFI-2 + FFPI for one result table -> dict code->mean."""
    return {c: float(np.mean([tbl[t][c][0] for t in TESTS])) for c in CODES}

def close(vals):
    """Repeat the first value at the end so the radar polygon closes."""
    return list(vals) + [vals[0]]

def make_radar(outfile):
    base_tbl = load("baseline")
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
            t = load(f"user_{u}_beta{b}")
            if t:
                per_user.append(avg_over_tests(t))
        if per_user:
            dpo[b] = {c: float(np.mean([pu[c] for pu in per_user])) for c in CODES}
            print(f"  beta {b}: averaged {len(per_user)}/{len(USERS)} users", flush=True)

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

    # fixed radial range shared with allusers_radar_100.png, so the two plots are
    # directly comparable. Union of every value in EITHER plot: target + baseline,
    # the per-user-averaged DPO lines, and the all-users (pooled) lines.
    allu = {b: avg_over_tests(load(f"ALLUSERS_beta{b}")) for b in BETAS
            if load(f"ALLUSERS_beta{b}")}
    allv = [v for c in CODES for v in
            ([target[c], baseline[c]]
             + [dpo[b][c] for b in BETAS if b in dpo]
             + [allu[b][c] for b in BETAS if b in allu])]
    lo = max(1.0, np.floor((min(allv) - 0.05) * 20) / 20)
    hi = min(5.0, np.ceil((max(allv) + 0.05) * 20) / 20)
    ax.set_xticks(angles); ax.set_xticklabels(NAMES)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.round(np.linspace(lo, hi, 5), 2))
    ax.tick_params(axis="y", labelsize=8); ax.set_rlabel_position(90)
    ax.set_title(f"{SHORT}  -  OCEAN averaged over all {len(USERS)} users\n(radial axis zoomed to data)", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9)
    plt.tight_layout(); plt.savefig(outfile, dpi=150, bbox_inches="tight"); plt.close()
    return True

od = os.path.join(HERE, "g_plots", FAMILY, SHORT); os.makedirs(od, exist_ok=True)
out = os.path.join(od, "avg_users_radar_100.png")
if make_radar(out):
    print(f"RADAR_DONE  wrote {out}")
else:
    print(f"SKIP {SHORT}: no baseline")
