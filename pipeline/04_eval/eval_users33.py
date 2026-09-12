"""Eval the 60 user adapters (20 users x 3 betas) of WizardLM-33B-Uncensored.
Round-robin the variants across free GPUs; each worker loads the 33B once and
hot-swaps its share via reeval20.py. Resumable (reeval20 skips done results)."""
import os, sys, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "cognitivecomputations/WizardLM-33B-V1.0-Uncensored"
SHORT = "WizardLM-33B-Uncensored"
ADAP = f"/mnt/ssd3/user_adapters/{SHORT}"
BETAS = ["0.01", "0.1", "0.5"]
USERS = json.load(open(os.path.join(HERE, "user_ids.json")))

JOBS = []
for u in USERS:
    for b in BETAS:
        lab = f"user_{u}_beta{b}"
        JOBS.append({"label": lab, "adapter": os.path.join(ADAP, lab)})

todo = [j for j in JOBS if not os.path.exists(os.path.join(HERE, f"results20_{SHORT}_{j['label']}.json"))]
print(f"eval: {len(todo)}/{len(JOBS)} user variants to score", flush=True)
if not todo:
    print("EVAL33_USERS_DONE", flush=True); sys.exit(0)

def free_gpus():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    return [int(l.split(",")[0]) for l in out
            if int(l.split(",")[1]) > 25000 and int(l.split(",")[2]) < 50]

gpus = free_gpus()
while not gpus:
    time.sleep(30); gpus = free_gpus()
print(f"free GPUs: {gpus}", flush=True)

groups = {g: [] for g in gpus}
for i, j in enumerate(todo):
    groups[gpus[i % len(gpus)]].append(j)

procs = []
for g, js in groups.items():
    if not js: continue
    jf = os.path.join(HERE, f"jobs33u_gpu{g}.json"); json.dump(js, open(jf, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    log = os.path.join(HERE, f"eval33u_gpu{g}.log")
    p = subprocess.Popen(["python3", "-u", "reeval20.py", BASE, SHORT, jf],
                         env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    procs.append(p)
    print(f"  GPU {g}: {len(js)} variants pid {p.pid}", flush=True)
    time.sleep(20)

for p in procs: p.wait()
print("EVAL33_USERS_DONE", flush=True)
