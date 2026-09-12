"""Extract per-example residual-stream difference vectors for (a) SYCOPHANCY
(matches-user answer minus honest answer) and (b) all 5 OCEAN traits (high minus
low), on WizardLM-13B-V1.2, so we can compare their directions / subspaces.
Sycophancy contrast cancels A/B letter identity because answer_matching_behavior
is randomized across A and B (the semantic 'be sycophantic' axis survives).
Usage: python extract_syc_traits.py <device> [n_syc]
"""
import os, sys, torch
os.environ["HF_HOME"] = "/mnt/ssd3/hf_cache"          # must precede HF imports
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from datasets import load_dataset
from mi_common import load_model, vicuna, trait_pairs, TRAITS

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
N_SYC = int(sys.argv[2]) if len(sys.argv) > 2 else 160
model, tok = load_model(DEV)
print("model loaded", flush=True)

@torch.no_grad()
def last_hidden(text):
    inp = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(DEV)
    out = model(**inp, output_hidden_states=True)
    return torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])  # [L+1, H]

# ---- sycophancy (BALANCED across A/B matching so letter identity cancels) ----
d = load_dataset("EleutherAI/sycophancy", "sycophancy_on_political_typology_quiz", split="validation")
half = N_SYC // 2
syc, nA, nB, i = [], 0, 0, 0
while (nA < half or nB < half) and i < len(d):
    ex = d[i]; i += 1
    isA = "(A)" in ex["answer_matching_behavior"]
    if (isA and nA >= half) or ((not isA) and nB >= half):
        continue
    q = vicuna(ex["question"])
    sm = last_hidden(q + ex["answer_matching_behavior"])      # sycophantic (agrees with user)
    sn = last_hidden(q + ex["answer_not_matching_behavior"])  # honest
    syc.append(sm - sn)
    nA += isA; nB += (not isA)
syc = torch.stack(syc)   # [~N_SYC, L+1, H]
print(f"sycophancy: {len(syc)} pairs | A-matching {nA}, B-matching {nB} (balanced)", flush=True)

# ---- traits (per-example high-low) ----
trait_diffs = {}
for T in TRAITS:
    ds = []
    for (ht, lt) in trait_pairs(T):
        ds.append(last_hidden(ht) - last_hidden(lt))
    trait_diffs[T] = torch.stack(ds)   # [60, L+1, H]
    print(f"trait {T}: {len(ds)} pairs", flush=True)

torch.save({"syc_diffs": syc, "trait_diffs": trait_diffs}, "syc_traits_diffs.pt")
print("saved syc_traits_diffs.pt", flush=True)
