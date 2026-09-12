"""Exp 2b: train a sparse autoencoder on the collected residual activations,
then find the dictionary features that distinguish high vs low for each OCEAN
trait. Reports top trait features and their decoder-direction cosine with the
Exp1 steering directions (if present).
Usage: python sae_train.py <device>
"""
import sys, json, torch, torch.nn as nn
import numpy as np
from mi_common import TRAITS, TRAIT_NAME

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
blob = torch.load("sae_acts.pt")
X = blob["X_train"].to(DEV)
Xlab = blob["X_lab"].to(DEV)
labels = blob["labels"]; LAYER = blob["layer"]
N, H = X.shape
N_FEAT = 8192
L1 = 4e-3
STEPS = 6000
BS = 4096
print(f"training SAE: N={N} hidden={H} feats={N_FEAT} layer={LAYER}", flush=True)

# normalize inputs (center + scale to unit mean-norm)
mu = X.mean(0, keepdim=True)
Xc = X - mu
scale = Xc.norm(dim=-1).mean()
Xc = Xc / scale

class SAE(nn.Module):
    def __init__(self, h, f):
        super().__init__()
        self.enc = nn.Linear(h, f, bias=True)
        self.dec = nn.Linear(f, h, bias=False)
        with torch.no_grad():
            self.dec.weight.div_(self.dec.weight.norm(dim=0, keepdim=True) + 1e-6)
            self.enc.weight.copy_(self.dec.weight.t())
    def forward(self, x):
        a = torch.relu(self.enc(x))
        return self.dec(a), a

sae = SAE(H, N_FEAT).to(DEV).float()
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
for step in range(STEPS):
    idx = torch.randint(0, N, (BS,), device=DEV)
    x = Xc[idx]
    recon, a = sae(x)
    mse = (recon - x).pow(2).sum(-1).mean()
    l1 = a.abs().sum(-1).mean()
    loss = mse + L1 * l1
    opt.zero_grad(); loss.backward()
    with torch.no_grad():  # keep decoder columns unit-norm
        sae.dec.weight.div_(sae.dec.weight.norm(dim=0, keepdim=True) + 1e-6)
    opt.step()
    if step % 1000 == 0:
        l0 = (a > 0).float().sum(-1).mean()
        print(f"  step {step}  mse {mse.item():.3f}  L0 {l0.item():.1f}", flush=True)

# feature activations on the labeled set
with torch.no_grad():
    _, A = sae((Xlab - mu) / scale)            # [M, n_feat]
A = A.cpu()
labels = np.array(labels, dtype=object)
traits = np.array([l[0] for l in blob["labels"]])
signs = np.array([l[1] for l in blob["labels"]])

# Exp1 directions for cosine
DIRS = None
try:
    DIRS = torch.load("steer_dirs.pt")["dirs"]
except Exception:
    pass

report = {"layer": LAYER, "traits": {}}
for T in TRAITS:
    hi = A[(traits == T) & (signs == 1)].mean(0)
    lo = A[(traits == T) & (signs == -1)].mean(0)
    diff = (hi - lo)
    top = torch.topk(diff.abs(), 8).indices.tolist()
    feats = []
    for fi in top:
        entry = {"feature": int(fi), "high_act": round(float(hi[fi]), 3),
                 "low_act": round(float(lo[fi]), 3), "diff": round(float(diff[fi]), 3)}
        if DIRS is not None:
            dvec = DIRS[T][LAYER + 1]
            dcol = sae.dec.weight[:, fi].detach().cpu()
            entry["cos_vs_dir"] = round(float(torch.nn.functional.cosine_similarity(
                dcol, dvec / (dvec.norm() + 1e-6), dim=0).abs()), 3)
        feats.append(entry)
    report["traits"][T] = feats
    print(f"\n[{TRAIT_NAME[T]}] top features (high-low):", flush=True)
    for e in feats:
        print(f"   feat {e['feature']:5d}  hi {e['high_act']:.2f} lo {e['low_act']:.2f} "
              f"Δ {e['diff']:+.2f}" + (f"  cos|dir| {e.get('cos_vs_dir')}" if 'cos_vs_dir' in e else ""), flush=True)

torch.save(sae.state_dict(), "sae_model.pt")
json.dump(report, open("sae_report.json", "w"), indent=2)
print("\nsaved sae_model.pt + sae_report.json", flush=True)
