"""Exp 3a: analyze the LoRA weight deltas of the 13B agent adapters to localize
*where* the personality shift is written into the weights. For each adapter the
effective delta is (alpha/r) * B @ A per target projection. We report the
Frobenius norm of that delta per layer/projection, and -- if Exp1 directions
exist -- the cosine between the delta's top output singular vector and the
steering direction at that layer.
Pure CPU (loads only the small adapter tensors). Fast."""
import os, glob, json
import numpy as np
import torch
from safetensors.torch import load_file

ADAP_DIR = "adapters_WizardLM-13B-V1.2"
ADAPTERS = ["high_beta0.01", "low_beta0.01"]

def load_adapter(name):
    d = os.path.join(ADAP_DIR, name)
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    r = cfg["r"]; alpha = cfg["lora_alpha"]; scale = alpha / r
    sf = glob.glob(os.path.join(d, "adapter_model.safetensors"))
    sd = load_file(sf[0]) if sf else torch.load(os.path.join(d, "adapter_model.bin"), map_location="cpu")
    # group A/B by module key
    mods = {}
    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k:
            base = k.split(".lora_")[0]
            mods.setdefault(base, {})["A" if "lora_A" in k else "B"] = v.float()
    return mods, scale

# try to attach Exp1 directions for cosine alignment
DIRS = None
if os.path.exists("steer_dirs.pt"):
    DIRS = torch.load("steer_dirs.pt")["dirs"]

def layer_of(modkey):
    # ...model.layers.<L>.self_attn.<proj>_proj...
    for part in modkey.split("."):
        if part.isdigit():
            return int(part)
    return -1

def proj_of(modkey):
    for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if p in modkey:
            return p
    return "other"

report = {}
for name in ADAPTERS:
    mods, scale = load_adapter(name)
    per_layer = {}
    per_proj = {}
    rows = []
    for mk, ab in mods.items():
        if "A" not in ab or "B" not in ab:
            continue
        delta = scale * (ab["B"] @ ab["A"])          # [out, in]
        fro = float(delta.norm())
        L = layer_of(mk); P = proj_of(mk)
        per_layer[L] = per_layer.get(L, 0.0) + fro**2
        per_proj[P] = per_proj.get(P, 0.0) + fro**2
        rows.append((L, P, fro, delta))
    per_layer = {L: float(np.sqrt(v)) for L, v in per_layer.items()}
    per_proj = {P: float(np.sqrt(v)) for P, v in per_proj.items()}
    order = sorted(per_layer, key=lambda L: -per_layer[L])
    report[name] = {"per_layer": per_layer, "per_proj": per_proj, "top_layers": order[:8]}
    print(f"\n=== {name} ===", flush=True)
    print("  delta Frobenius by projection:", {k: round(v, 2) for k, v in per_proj.items()}, flush=True)
    print("  top-8 layers by delta magnitude:", [(L, round(per_layer[L], 2)) for L in order[:8]], flush=True)

    # cosine: top output singular vector of each o_proj delta vs Agreeableness direction
    if DIRS is not None:
        for (L, P, fro, delta) in rows:
            if P != "o_proj":
                continue
            U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
            top_out = U[:, 0]                         # principal output direction [hidden]
            cosines = {}
            for T, dvec in DIRS.items():
                dl = dvec[L + 1]
                cosines[T] = round(float(torch.nn.functional.cosine_similarity(
                    top_out, dl / (dl.norm() + 1e-6), dim=0).abs()), 3)
            if L in order[:5]:
                print(f"    L{L} o_proj top-SV |cos| vs trait dirs: {cosines}", flush=True)

json.dump(report, open("lora_delta_report.json", "w"), indent=2)
print("\nsaved lora_delta_report.json", flush=True)
