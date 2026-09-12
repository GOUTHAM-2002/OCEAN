"""
Radar chart of the OCEAN profile of the POOLED "all-users" DPO models for
vicuna-7b-v1.5, over the 100 current users. Adapted from radar_allusers_100_7b.py.

Uses the 3 single models each DPO-trained on ALL 100 users' combined
preferences at one beta:
  results_vicuna-7b-v1.5_ALLUSERS_beta{0.01,0.1,0.5}.json
For each beta the model OCEAN is the mean over BFI-2 + FFPI -> one line per beta.
The baseline (SFT) line and the users' actual-OCEAN average (target) are drawn
for reference. The radial axis range is computed from the union of all values
in this plot plus the per-user DPO averages, so it is comparable to a per-user
averaged radar built the same way.

Writes: g_plots/Vicuna/vicuna-7b-v1.5/allusers_radar_100.png
Usage (after the sweep): python3 radar_allusers_100_vicuna7b.py
"""
import os, json
import numpy as np, pandas as pd
if not hasattr(np, "float"):       # old matplotlib polar code references the removed np.float alias
    np.float = float
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODES = ["O", "C", "E", "A", "N"]
NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TESTS = ["BFI-2", "FFPI"]
BETAS = ["0.01", "0.1", "0.5"]
FAMILY, SHORT = "Vicuna", "vicuna-7b-v1.5"
BETA_COLORS = {"0.01": "#4C72B0", "0.1": "#55A868", "0.5": "#C44E52"}
USERS = json.load(open(os.path.join(HERE, "user_ids_100.json")))

# users' actual OCEAN (1-5), scaled from the 60-point IPIP-300 trait scores
resp = pd.read_csv(os.path.join(HERE, "ipip300_responses_and_scores.csv"))
resp["case_id"] = resp["case_id"].astype(str)
SC = {"O": "Openness_score", "C": "Conscientiousness_score", "E": "Extraversion_score",
      "A": "Agreeableness_score", "N": "Neuroticism_score"}
USER_SCORES = {int(u): {c: float(resp.loc[resp.case_id == str(u), col].iloc[0]) / 60
                        for c, col in SC.items()} for u in USERS}

def load(label):
    p = os.path.join(HERE, f"results_{SHORT}_{label}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None

def avg_over_tests(tbl):
    """OCEAN mean over BFI-2 + FFPI for one result table -> dict code->mean."""
    return {c: float(np.mean([tbl[t][c][0] for t in TESTS])) for c in CODES}

def close(vals):
    """Repeat the first value at the end so the radar polygon closes."""
    return list(vals) + [vals[0]]

def shared_range():
    """Fixed [lo, hi] over every value in either the per-user-averaged or the
    all-users plot: target + baseline, the per-user-averaged DPO lines, and the
    all-users (pooled) lines. Same padding formula as the wizardLM version."""
    base_tbl = load("baseline")
    baseline = avg_over_tests(base_tbl)
    target = {c: float(np.mean([USER_SCORES[int(u)][c] for u in USERS])) for c in CODES}

    dpo = {}
    for b in BETAS:
        per_user = []
        for u in USERS:
            t = load(f"user_{u}_beta{b}")
            if t:
                per_user.append(avg_over_tests(t))
        if per_user:
            dpo[b] = {c: float(np.mean([pu[c] for pu in per_user])) for c in CODES}

    allu = {b: avg_over_tests(load(f"ALLUSERS_beta{b}")) for b in BETAS
            if load(f"ALLUSERS_beta{b}")}

    allv = [v for c in CODES for v in
            ([target[c], baseline[c]]
             + [dpo[b][c] for b in BETAS if b in dpo]
             + [allu[b][c] for b in BETAS if b in allu])]
    lo = max(1.0, np.floor((min(allv) - 0.05) * 20) / 20)
    hi = min(5.0, np.ceil((max(allv) + 0.05) * 20) / 20)
    return lo, hi

def make_radar(outfile):
    base_tbl = load("baseline")
    if not base_tbl:
        return False
    baseline = avg_over_tests(base_tbl)

    # target: mean actual OCEAN across all users
    target = {c: float(np.mean([USER_SCORES[int(u)][c] for u in USERS])) for c in CODES}

    # all-users (pooled) models: OCEAN mean over BFI-2 + FFPI, per beta
    allu = {}
    for b in BETAS:
        t = load(f"ALLUSERS_beta{b}")
        if t:
            allu[b] = avg_over_tests(t)
            print(f"  beta {b}: loaded all-users model", flush=True)

    angles = np.linspace(0, 2 * np.pi, len(CODES), endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)

    for b in BETAS:
        if b in allu:
            vals = close([allu[b][c] for c in CODES])
            ax.plot(angles_c, vals, color=BETA_COLORS[b], lw=2, marker="o", ms=4,
                    label=f"All-users DPO (beta={b})")
            ax.fill(angles_c, vals, color=BETA_COLORS[b], alpha=0.15)
    base_vals = close([baseline[c] for c in CODES])
    ax.plot(angles_c, base_vals, color="#7f7f7f", lw=2, marker="o", ms=4, label="Baseline (SFT)")
    ax.fill(angles_c, base_vals, color="#7f7f7f", alpha=0.12)
    tgt_vals = close([target[c] for c in CODES])
    ax.fill(angles_c, tgt_vals, color="black", alpha=0.06)
    ax.plot(angles_c, tgt_vals, ls="--", color="black", lw=3, marker="D", ms=6,
            zorder=10, label="User actual (avg)")

    lo, hi = shared_range()
    ax.set_xticks(angles); ax.set_xticklabels(NAMES)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.round(np.linspace(lo, hi, 5), 2))
    ax.tick_params(axis="y", labelsize=8); ax.set_rlabel_position(90)
    ax.set_title(f"{SHORT}  -  all-users DPO model vs average user OCEAN", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9)
    plt.tight_layout(); plt.savefig(outfile, dpi=150, bbox_inches="tight"); plt.close()
    return True

od = os.path.join(HERE, "g_plots", FAMILY, SHORT); os.makedirs(od, exist_ok=True)
out = os.path.join(od, "allusers_radar_100.png")
if make_radar(out):
    print(f"RADAR_DONE  wrote {out}")
else:
    print(f"SKIP {SHORT}: no baseline")
