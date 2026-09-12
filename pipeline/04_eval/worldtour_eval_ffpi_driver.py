"""Split the 200 last-dance adapters across all free GPUs and launch eval workers."""
import os, json, subprocess, time

IDS = [str(k) for k in sorted(int(k) for k in json.load(open("worldtour_meta.json")))]
todo = [c for c in IDS if not os.path.exists(f"results_worldtour_ffpi/{c}.json")
        and os.path.exists(f"adapters_worldtour/{c}/adapter_config.json")]
print(f"eval: {len(todo)} adapters to score")

def free_gpus():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    return [int(l.split(",")[0]) for l in out if int(l.split(",")[1]) > 38000 and int(l.split(",")[2]) < 30]

gpus = free_gpus()
while not gpus: time.sleep(30); gpus = free_gpus()
print("free GPUs:", gpus)
groups = {g: [] for g in gpus}
for i, c in enumerate(todo): groups[gpus[i % len(gpus)]].append(c)

procs = []
os.makedirs("worldtour_logs", exist_ok=True)
for g, cs in groups.items():
    if not cs: continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # worker sets it
    log = open(f"worldtour_logs/eval_ffpi_gpu{g}.log", "w")
    p = subprocess.Popen(["python3", "-u", "worldtour_eval_ffpi.py", str(g), ",".join(cs)],
                         stdout=log, stderr=subprocess.STDOUT)
    procs.append(p); print(f"  GPU{g}: {len(cs)} adapters pid {p.pid}"); time.sleep(15)
for p in procs: p.wait()
print("LASTDANCE_EVAL_ALL_DONE")
