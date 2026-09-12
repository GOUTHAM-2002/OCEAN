"""Phase-A plots: AGENT_LOW/HIGH (3 bars) + macro_delta_summary.
Usage: python wiz_agent_plots.py <SHORT>   (e.g. WizardLM-33B-Uncensored)"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = sys.argv[1] if len(sys.argv) > 1 else "WizardLM-33B-Uncensored"
OUTD = os.path.join(HERE, "g_plots", "WizardLM", SHORT)
CODES = ["O", "C", "E", "A", "N"]
NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TESTS = ["BFI-2", "FFPI"]; SERIES = ["Baseline (SFT)"] + TESTS
COLORS = ["#7f7f7f", "#4C72B0", "#C44E52"]; BETAS = ["0.01", "0.1", "0.5"]

def load(label):
    p = os.path.join(HERE, f"results20_{SHORT}_{label}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None

base = load("baseline")

def make_plot(dpo, outfile):
    b = {c: [np.mean([base[t][c][0] for t in TESTS]), np.mean([base[t][c][1] for t in TESTS])] for c in CODES}
    x = np.arange(5); w = 0.26; fig, ax = plt.subplots(figsize=(10, 5.5))
    for si, s in enumerate(SERIES):
        ms = []; cs = []
        for c in CODES:
            m, ci = b[c] if s == "Baseline (SFT)" else dpo[s][c]; ms.append(m); cs.append(ci)
        ax.bar(x + (si - 1) * w, ms, w, yerr=cs, capsize=4, label=s, color=COLORS[si], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("Score (1-5)"); ax.set_ylim(1, 5); ax.legend()
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

for a, folder in [("low", "AGENT_LOW"), ("high", "AGENT_HIGH")]:
    od = os.path.join(OUTD, folder); os.makedirs(od, exist_ok=True)
    for b in BETAS:
        t = load(f"{a}_beta{b}")
        if t: make_plot(t, os.path.join(od, f"{folder.lower()}_beta{b}.png"))

# macro_delta_summary (grey, minimal, same style as 13B)
def macro(label):
    t = load(label)
    return np.mean([np.mean([t[ti][c][0] for ti in TESTS]) for c in CODES]) if t else None

bm = macro("baseline")
rows = [(f"{d.upper()} β={b}", macro(f"{d}_beta{b}") - bm)
        for d in ["high", "low"] for b in BETAS if macro(f"{d}_beta{b}") is not None]
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(10, 5)); y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], color="#c8c8c8", edgecolor="black", height=0.62)
ax.axvline(0, color="black", lw=1.0); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
mx = max(abs(r[1]) for r in rows) * 1.25; ax.set_xlim(-mx, mx)
ax.set_xlabel("Average OCEAN score shift", fontsize=12); ax.set_title(SHORT, fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(OUTD, "macro_delta_summary.png"), dpi=150); plt.close()
print("Phase-A plots written ->", OUTD)
