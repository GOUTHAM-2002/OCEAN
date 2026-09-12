"""Summary figures for the mechanistic-interpretability experiments on
WizardLM-13B-V1.2. Reads steer_result_*.json (Exp1), patch_result.json (Exp3b),
lora_delta_report.json (Exp3a). Writes to g_plots/mech_interp/."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "g_plots/mech_interp"; os.makedirs(OUT, exist_ok=True)
TRAITS = ["O", "C", "E", "A", "N"]
NAME = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
        "A": "Agreeableness", "N": "Neuroticism"}

# ---- Exp1: steering causal curves (one panel per trait, line per layer) ----
have = [T for T in TRAITS if os.path.exists(f"steer_result_{T}.json")]
if have:
    fig, axes = plt.subplots(1, len(have), figsize=(4*len(have), 4), squeeze=False)
    for ax, T in zip(axes[0], have):
        r = json.load(open(f"steer_result_{T}.json")); base = r["baseline"]
        for g in r["grid"]:
            cs = [c for c, s in g["curve"]]; ss = [s for c, s in g["curve"]]
            ax.plot(cs, ss, marker="o", label=f"layer {g['layer']}")
        ax.axhline(base, ls="--", color="gray", lw=1)
        ax.set_title(NAME[T]); ax.set_xlabel("steering coefficient"); ax.set_ylabel("trait score (1-5)")
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/exp1_steering_curves.png", dpi=150); plt.close()
    print(f"wrote exp1_steering_curves.png ({len(have)} traits)")

# ---- Exp3b: patch recovery vs layer ----
if os.path.exists("patch_result.json"):
    p = json.load(open("patch_result.json"))
    rec = p["recovery"]; shift = p["shift"]
    L = [r["layer"] for r in rec]; frac = [100*r["recovered"]/shift if shift else 0 for r in rec]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(L, frac, color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("patched layer"); ax.set_ylabel("% of adapter shift recovered")
    ax.set_title("Exp3b: where the high-agent shift is causally written (activation patching)")
    plt.tight_layout(); plt.savefig(f"{OUT}/exp3b_patch_recovery.png", dpi=150); plt.close()
    print("wrote exp3b_patch_recovery.png")

# ---- Exp3a: LoRA weight-delta magnitude per layer (HIGH vs LOW) ----
if os.path.exists("lora_delta_report.json"):
    d = json.load(open("lora_delta_report.json"))
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, color in [("high_beta0.01", "#C44E52"), ("low_beta0.01", "#55A868")]:
        if name in d:
            pl = d[name]["per_layer"]
            xs = sorted(int(k) for k in pl)
            ax.plot(xs, [pl[str(x)] for x in xs], marker="o", ms=3, label=name, color=color)
    ax.set_xlabel("layer"); ax.set_ylabel("LoRA delta Frobenius norm")
    ax.set_title("Exp3a: where each adapter changes weights (HIGH=late, LOW=middle)")
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/exp3a_lora_delta.png", dpi=150); plt.close()
    print("wrote exp3a_lora_delta.png")

print("done ->", OUT)
