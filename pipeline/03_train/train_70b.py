"""
Train QLoRA-DPO adapters for WizardLM-70B-V1.0.
Usage: python train_70b.py <agents|users> <gpus_per_job>
Resumable (skips adapters already saved). Pins gpus_per_job GPUs per training.
"""
import os, sys, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
WHICH = sys.argv[1]
GPER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MODEL = sys.argv[3] if len(sys.argv) > 3 else "WizardLMTeam/WizardLM-70B-V1.0"
SHORT = sys.argv[4] if len(sys.argv) > 4 else "WizardLM-70B-V1.0"
BETAS = ["0.01", "0.1", "0.5"]
ADAP_AGENTS = os.path.join(HERE, f"adapters_{SHORT}")
ADAP_USERS = f"/mnt/ssd3/user_adapters/{SHORT}"
os.makedirs(ADAP_AGENTS, exist_ok=True); os.makedirs(ADAP_USERS, exist_ok=True)

def free_gpus():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    return [int(l.split(",")[0]) for l in out
            if int(l.split(",")[1]) > 30000 and int(l.split(",")[2]) < 50]

if WHICH == "agents":
    JOBS = [(f"{a}_beta{b}", os.path.join(HERE, f"dpo_agent_{a}.jsonl"), b,
             os.path.join(ADAP_AGENTS, f"{a}_beta{b}")) for a in ["low", "high"] for b in BETAS]
else:
    USERS = json.load(open(os.path.join(HERE, "user_ids.json")))
    JOBS = [(f"user_{u}_beta{b}", os.path.join(HERE, f"user_dpo/dpo_user_{u}.jsonl"), b,
             os.path.join(ADAP_USERS, f"user_{u}_beta{b}")) for u in USERS for b in BETAS]

todo = [j for j in JOBS if not os.path.exists(os.path.join(j[3], "adapter_config.json"))]
print(f"{WHICH}: {len(todo)}/{len(JOBS)} adapters to train, {GPER} GPU(s)/job", flush=True)

gpus = free_gpus()
while len(gpus) < GPER:
    time.sleep(30); gpus = free_gpus()
slots = [gpus[i:i+GPER] for i in range(0, len(gpus) - GPER + 1, GPER)]  # non-overlapping GPU groups
print(f"GPU slots: {slots}", flush=True)

running = {}  # popen -> slot
queue = list(todo); free_slots = list(slots)
while queue or running:
    while queue and free_slots:
        lab, data, beta, out = queue.pop(0); slot = free_slots.pop(0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, slot)))
        log = os.path.join(HERE, f"tr70_{WHICH}_{lab}.log")
        p = subprocess.Popen(["python3", "-u", "dpo_train.py", MODEL, data, beta, out],
                             env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        running[p] = slot
        print(f"  train {lab} -> GPUs {slot} pid {p.pid}", flush=True)
        time.sleep(90)   # stagger heavy 70B loads
    time.sleep(15)
    for p in list(running):
        if p.poll() is not None:
            slot = running.pop(p); free_slots.append(slot)
            print(f"  [rc={p.returncode}] slot {slot} freed", flush=True)
print(f"TRAIN70_{WHICH.upper()}_DONE", flush=True)
