"""Train 200 per-person DPO adapters (100 male + 100 female) on WizardLM-13B-V1.2,
beta=0.01. Dynamic scheduler: grabs any GPU that is truly free (>38GB free, low
util) so it coexists with the finishing Phase-C run and expands as GPUs free up.
Resumable: skips adapters already saved.
"""
import os, sys, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "WizardLMTeam/WizardLM-13B-V1.2"
BETA = "0.01"
OUT = os.path.join(HERE, "adapters_lastdance"); os.makedirs(OUT, exist_ok=True)
IDS = json.load(open("ld_males.json")) + json.load(open("ld_females.json"))

def free_gpus():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    return [int(l.split(",")[0]) for l in out
            if int(l.split(",")[1]) > 38000 and int(l.split(",")[2]) < 30]

jobs = [cid for cid in IDS
        if not os.path.exists(os.path.join(OUT, str(cid), "adapter_config.json"))]
print(f"to train: {len(jobs)}/{len(IDS)} adapters", flush=True)

running = {}   # popen -> (gpu, cid)
queue = list(jobs)
busy_gpus = set()
while queue or running:
    fg = [g for g in free_gpus() if g not in busy_gpus]
    while queue and fg:
        cid = queue.pop(0); gpu = fg.pop(0); busy_gpus.add(gpu)
        data = os.path.join(HERE, "user_dpo", f"dpo_user_{cid}.jsonl")
        outd = os.path.join(OUT, str(cid))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   HF_HOME="/mnt/ssd3/hf_cache", HF_HUB_OFFLINE="1")
        log = os.path.join(HERE, "lastdance_logs"); os.makedirs(log, exist_ok=True)
        p = subprocess.Popen(["python3", "-u", "dpo_train.py", MODEL, data, BETA, outd],
                             env=env, stdout=open(os.path.join(log, f"train_{cid}.log"), "w"),
                             stderr=subprocess.STDOUT)
        running[p] = (gpu, cid)
        print(f"  train {cid} -> GPU{gpu} ({len(jobs)-len(queue)}/{len(jobs)})", flush=True)
        time.sleep(20)   # stagger model loads
    time.sleep(15)
    for p in list(running):
        if p.poll() is not None:
            gpu, cid = running.pop(p); busy_gpus.discard(gpu)
            ok = os.path.exists(os.path.join(OUT, str(cid), "adapter_config.json"))
            print(f"  [{cid}] rc={p.returncode} saved={ok}; GPU{gpu} freed", flush=True)
            if not ok and cid not in queue:
                queue.append(cid)   # retry once at the end
print("LASTDANCE_TRAIN_DONE", flush=True)
