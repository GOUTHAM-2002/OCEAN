"""Exp 2a: collect residual-stream activations at one layer of WizardLM-13B-V1.2
over a personality corpus, for training a sparse autoencoder. Stores:
  - X_train [N, hidden]: all token positions (unlabeled) for SAE training
  - X_lab   [M, hidden] + labels (trait, sign): mean response activation per
    contrastive example, for trait-feature attribution.
Usage: python sae_collect.py <layer> <device>
"""
import sys, json, torch
from mi_common import load_model, trait_pairs, vicuna, TRAITS

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
model, tok = load_model(DEV)
print(f"model loaded; collecting at layer {LAYER}", flush=True)

@torch.no_grad()
def hiddens(text):
    inp = tok(text, return_tensors="pt").to(DEV)
    out = model(**inp, output_hidden_states=True)
    return out.hidden_states[LAYER + 1][0].float().cpu()   # [seq, hidden]

X_train, X_lab, labels = [], [], []
extra_corpus = []
# add questionnaire item statements as extra personality text
import pandas as pd
for qf in ["bfi2_questions.csv", "ffpi_questions.csv", "ipip300_questions.csv"]:
    try:
        q = pd.read_csv(qf)
        extra_corpus += [vicuna(f'Statement: "{t}"') for t in q["question_text"].tolist()]
    except Exception:
        pass

for T in TRAITS:
    for (ht, lt) in trait_pairs(T):
        for text, sign in [(ht, +1), (lt, -1)]:
            h = hiddens(text)                 # [seq, hidden]
            X_train.append(h)
            X_lab.append(h[-1])               # answer token = where the trait is expressed
            labels.append((T, sign))

for text in extra_corpus:
    X_train.append(hiddens(text))

X_train = torch.cat(X_train, 0)
X_lab = torch.stack(X_lab)
torch.save({"X_train": X_train, "X_lab": X_lab, "labels": labels, "layer": LAYER},
           "sae_acts.pt")
print(f"saved sae_acts.pt | train {tuple(X_train.shape)} | labeled {tuple(X_lab.shape)}", flush=True)
