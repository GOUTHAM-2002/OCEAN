"""
Drive the 20-seed re-eval (BFI-2 + FFPI, Goldberg dropped) across all variants:
Mixtral (7) + 6 small models (67 each). For each base model, split its variants
across 4 GPUs; each worker loads the base once and hot-swaps adapters. Resumable.
"""
import os, json, subprocess, math, time

HERE = os.path.dirname(os.path.abspath(__file__))
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
# Mixtral 47B in 4-bit holds tens of GB CPU RAM per process -> running 4 at once
# OOM-kills them. Cap concurrency for the big model; small 7-13B models are fine at 4.
NWORKERS = {"mixtral": 2}
BETAS = ["0.01", "0.1", "0.5"]
USERS = json.load(open(os.path.join(HERE, "user_ids.json")))
SMALL = {"tulu-2-7b": "allenai/tulu-2-7b", "tulu-2-13b": "allenai/tulu-2-13b",
         "vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5", "vicuna-13b-v1.5": "lmsys/vicuna-13b-v1.5",
         "wizardLM-7B": "TheBloke/wizardLM-7B-HF", "WizardLM-13B-V1.2": "WizardLMTeam/WizardLM-13B-V1.2"}

def small_variants(short):
    vs = [("baseline", "none")]
    for a in ["low", "high"]:
        for b in BETAS:
            vs.append((f"{a}_beta{b}", os.path.join(HERE, f"adapters_{short}/{a}_beta{b}")))
    for uid in USERS:
        for b in BETAS:
            vs.append((f"user_{uid}_beta{b}", f"/mnt/ssd3/user_adapters/{short}/user_{uid}_beta{b}"))
    return vs

BASES = [("mixtral", "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
          [("baseline", "none")] +
          [(f"{a}_beta{b}", os.path.join(HERE, f"adapters/{a}_beta{b}")) for a in ["low", "high"] for b in BETAS])]
for short, bid in SMALL.items():
    BASES.append((short, bid, small_variants(short)))

for short, bid, variants in BASES:
    todo = [(lab, ad) for (lab, ad) in variants
            if not os.path.exists(os.path.join(HERE, f"results20_{short}_{lab}.json"))]
    print(f"\n######## {short}: {len(todo)}/{len(variants)} variants to do ########", flush=True)
    if not todo:
        continue
    nchunks = min(NWORKERS.get(short, len(GPUS)), len(todo))
    chunks = [[] for _ in range(nchunks)]
    for i, v in enumerate(todo):
        chunks[i % nchunks].append({"label": v[0], "adapter": v[1]})
    procs = []
    for ci, chunk in enumerate(chunks):
        jf = os.path.join(HERE, f"jobs20_{short}_{ci}.json")
        json.dump(chunk, open(jf, "w"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPUS[ci]))
        log = os.path.join(HERE, f"re20_{short}_{ci}.log")
        p = subprocess.Popen(["python3", "-u", "reeval20.py", bid, short, jf],
                             env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        procs.append(p)
        print(f"  chunk {ci} ({len(chunk)} variants) -> GPU{GPUS[ci]} pid {p.pid}", flush=True)
        time.sleep(45)   # stagger so concurrent model loads don't spike CPU RAM together
    for p in procs:
        p.wait()
    print(f"######## {short} DONE ########", flush=True)

print("REEVAL20_ALL_DONE", flush=True)
