"""Eval baseline + 6 agent adapters of WizardLM-33B-Uncensored.
Splits the 7 variants across free GPUs; each worker loads the 33B once and
hot-swaps its share via reeval20.py. Resumable (reeval20 skips done results)."""
import os, sys, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "cognitivecomputations/WizardLM-33B-V1.0-Uncensored"
SHORT = "WizardLM-33B-Uncensored"
ADAP = os.path.join(HERE, f"adapters_{SHORT}")
BETAS = ["0.01", "0.1", "0.5"]

JOBS = [{"label": "baseline", "adapter": "none"}]
for a in ["low", "high"]:
    for b in BETAS:
        JOBS.append({"label": f"{a}_beta{b}", "adapter": os.path.join(ADAP, f"{a}_beta{b}")})

# only keep jobs whose results don't yet exist
todo = [j for j in JOBS if not os.path.exists(os.path.join(HERE, f"results20_{SHORT}_{j['label']}.json"))]
print(f"eval: {len(todo)}/{len(JOBS)} variants to score", flush=True)
if not todo:
    print("EVAL33_AGENTS_DONE", flush=True); sys.exit(0)

def free_gpus():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    return [int(l.split(",")[0]) for l in out
            if int(l.split(",")[1]) > 25000 and int(l.split(",")[2]) < 50]

gpus = free_gpus()
while not gpus:
    time.sleep(30); gpus = free_gpus()
print(f"free GPUs: {gpus}", flush=True)

# round-robin split variants across GPUs
groups = {g: [] for g in gpus}
for i, j in enumerate(todo):
    groups[gpus[i % len(gpus)]].append(j)

procs = []
for g, js in groups.items():
    if not js: continue
    jf = os.path.join(HERE, f"jobs33_gpu{g}.json"); json.dump(js, open(jf, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    log = os.path.join(HERE, f"eval33_gpu{g}.log")
    p = subprocess.Popen(["python3", "-u", "reeval20.py", BASE, SHORT, jf],
                         env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    procs.append(p)
    print(f"  GPU {g}: {[j['label'] for j in js]} pid {p.pid}", flush=True)
    time.sleep(20)

for p in procs: p.wait()
print("EVAL33_AGENTS_DONE", flush=True)
