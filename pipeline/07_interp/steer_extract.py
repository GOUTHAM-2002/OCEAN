"""Exp 1a: extract per-trait difference-of-means directions in the residual
stream of WizardLM-13B-V1.2. For each OCEAN trait, contrast item-aligned
high-trait vs low-trait responses; the direction at each layer is
mean(high last-token hidden) - mean(low last-token hidden).
Saves steer_dirs.pt = {trait: tensor[n_layers+1, hidden]} plus per-layer norms."""
import sys, torch
from mi_common import load_model, trait_pairs, TRAITS

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
model, tok = load_model(DEV)
print("model loaded", flush=True)

@torch.no_grad()
def last_token_hiddens(text):
    inp = tok(text, return_tensors="pt").to(DEV)
    out = model(**inp, output_hidden_states=True)
    # tuple of (n_layers+1) tensors [1, seq, hidden]; take last token
    return torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])  # [L+1, hidden]

dirs, norms = {}, {}
all_hidden_norms = []   # collect per-layer residual norms across all texts for scaling
for T in TRAITS:
    pairs = trait_pairs(T)
    his, los = [], []
    for ht, lt in pairs:
        hh = last_token_hiddens(ht); ll = last_token_hiddens(lt)
        his.append(hh); los.append(ll)
        all_hidden_norms.append(hh.norm(dim=-1)); all_hidden_norms.append(ll.norm(dim=-1))
    H = torch.stack(his).mean(0)   # [L+1, hidden]
    Lo = torch.stack(los).mean(0)
    d = H - Lo
    dirs[T] = d
    norms[T] = d.norm(dim=-1)       # [L+1] per-layer direction magnitude
    print(f"[{T}] {len(pairs)} pairs | peak-direction layer {int(d.norm(dim=-1).argmax())} "
          f"|d|max {float(d.norm(dim=-1).max()):.2f}", flush=True)

resid_norm = torch.stack(all_hidden_norms).mean(0)   # [L+1] typical residual L2 norm per layer
torch.save({"dirs": dirs, "norms": norms, "resid_norm_ref": resid_norm}, "steer_dirs.pt")
print("saved steer_dirs.pt | ref resid norm mid-layer ~", float(resid_norm[len(resid_norm)//2]), flush=True)
