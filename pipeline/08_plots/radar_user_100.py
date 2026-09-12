"""
Per-user OCEAN radar charts for WizardLM-13B-V1.2, over the 100 current users.
Per-user version of radar_allusers_100.py / radar_avg_users_100.py.

Instead of ONE pooled/averaged radar, this writes ONE radar PER USER for the
100 users in user_ids_100.json. Each user's radar shows 5 polygons on the
O/C/E/A/N axes:
  - Baseline (SFT)                       : grey  (#7f7f7f), filled alpha ~0.12
  - DPO beta=0.01 / 0.1 / 0.5            : BETA_COLORS, filled alpha ~0.15
  - that user's ACTUAL OCEAN (target)    : bold black dashed line, on top
Each model's OCEAN value = mean over BFI-2 + FFPI (avg_over_tests). The user's
actual OCEAN (1-5) is scaled from the 60-point IPIP-300 trait scores.

Per-user DPO result files are loaded exactly like radar_allusers_100.py, via
load(f"user_{u}_beta{b}") -> results_<short>_user_<uid>_beta<b>.json (no "20").

RADIAL AXIS: ONE GLOBAL FIXED RANGE, shared across ALL 100 radars (NOT per-user
auto-zoom). lo/hi are computed ONCE from the union of EVERY value that appears
in ANY user's radar: every user's actual OCEAN (all 5 traits), the baseline, and
every per-user DPO beta value that exists. Same padding/rounding formula as
radar_allusers_100.py's shared_range():
    lo = max(1.0, floor((min - 0.05) * 20) / 20)
    hi = min(5.0, ceil ((max + 0.05) * 20) / 20)
The SAME lo/hi is applied to every user's plot so all 100 radars are directly
comparable.

Writes: g_plots/WizardLM/WizardLM-13B-V1.2/user_<uid>/user_<uid>_radar.png
(one per user; sits next to the existing user_<uid>_beta*.png bar plots).
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
FAMILY, SHORT = "WizardLM", "WizardLM-13B-V1.2"
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

def user_dpo(u):
    """Per-user DPO OCEAN per beta -> {beta: {code: val}}, only betas present."""
    out = {}
    for b in BETAS:
        t = load(f"user_{u}_beta{b}")
        if t:
            out[b] = avg_over_tests(t)
    return out

def global_range(base_ocean, dpo_by_user):
    """ONE fixed [lo, hi] shared across all 100 radars. Unions every value that
    appears in ANY user's radar: every user's actual OCEAN, the baseline, and
    every per-user DPO beta value that exists. Same padding formula as
    radar_allusers_100.py's shared_range()."""
    allv = []
    for c in CODES:
        allv.append(base_ocean[c])
        for u in USERS:
            allv.append(USER_SCORES[int(u)][c])
        for dpo in dpo_by_user.values():
            for b in dpo:
                allv.append(dpo[b][c])
    lo = max(1.0, np.floor((min(allv) - 0.05) * 20) / 20)
    hi = min(5.0, np.ceil((max(allv) + 0.05) * 20) / 20)
    return lo, hi

def make_radar(u, base_ocean, dpo, lo, hi, outfile):
    """One user's radar: baseline + present DPO betas + that user's actual OCEAN,
    on the global fixed radial range."""
    target = USER_SCORES[int(u)]

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
                    label=f"DPO (beta={b})")
            ax.fill(angles_c, vals, color=BETA_COLORS[b], alpha=0.15)
    base_vals = close([base_ocean[c] for c in CODES])
    ax.plot(angles_c, base_vals, color="#7f7f7f", lw=2, marker="o", ms=4, label="Baseline (SFT)")
    ax.fill(angles_c, base_vals, color="#7f7f7f", alpha=0.12)
    tgt_vals = close([target[c] for c in CODES])
    ax.fill(angles_c, tgt_vals, color="black", alpha=0.06)
    ax.plot(angles_c, tgt_vals, ls="--", color="black", lw=3, marker="D", ms=6,
            zorder=10, label="User actual")

    # global fixed radial range shared across all 100 radars (not per-user auto-zoom)
    ax.set_xticks(angles); ax.set_xticklabels(NAMES)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.round(np.linspace(lo, hi, 5), 2))
    ax.tick_params(axis="y", labelsize=8); ax.set_rlabel_position(90)
    ax.set_title(f"{SHORT}  -  user {u}  OCEAN (DPO vs actual)", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=9)
    plt.tight_layout(); plt.savefig(outfile, dpi=150, bbox_inches="tight"); plt.close()

base_tbl = load("baseline")
if not base_tbl:
    print(f"SKIP {SHORT}: no baseline")
    raise SystemExit
base_ocean = avg_over_tests(base_tbl)

# load every user's DPO up front so the global range covers all of them
dpo_by_user = {}
skipped = []
for u in USERS:
    dpo = user_dpo(u)
    if not dpo:
        skipped.append((u, "missing all three beta files"))
        print(f"  user {u}: SKIP (no beta files)", flush=True)
        continue
    dpo_by_user[u] = dpo

lo, hi = global_range(base_ocean, dpo_by_user)
print(f"global radial range: lo={lo}  hi={hi}", flush=True)

written = 0
for u in USERS:
    if u not in dpo_by_user:
        continue
    dpo = dpo_by_user[u]
    od = os.path.join(HERE, "g_plots", FAMILY, SHORT, f"user_{u}")
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"user_{u}_radar.png")
    make_radar(u, base_ocean, dpo, lo, hi, out)
    written += 1
    print(f"  user {u}: wrote radar ({len(dpo)}/{len(BETAS)} betas)", flush=True)

print(f"RADAR_DONE  wrote {written} per-user radars  (range lo={lo} hi={hi})")
if skipped:
    print(f"SKIPPED {len(skipped)} users:")
    for u, why in skipped:
        print(f"  user {u}: {why}")
