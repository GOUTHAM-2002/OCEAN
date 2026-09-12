"""
Downstream benchmark sweep: SFT baseline + 6 personality-DPO adapters
(AGENT_LOW/HIGH x beta{0.01,0.1,0.5}) on 7 benchmarks via lm-eval 0.4.7.
One variant per GPU, variants in parallel; resumable (skips finished invocations).
"""
import os, glob, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"
BNB = ("load_in_4bit=True,bnb_4bit_quant_type=nf4,"
       "bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True")
VARIANTS = [("baseline", None)] + [
    (lab, os.path.join(HERE, "adapters", lab))
    for lab in ["low_beta0.01", "low_beta0.1", "low_beta0.5",
                "high_beta0.01", "high_beta0.1", "high_beta0.5"]]
# (group_name, tasks, extra_args)
GROUPS = [
    ("mmlu",   "mmlu",                                   ["--num_fewshot", "5", "--limit", "25"]),
    ("hsbq",   "hellaswag,boolq",                        ["--limit", "1000"]),
    ("full",   "arc_challenge,truthfulqa_mc2,winogrande", []),
    ("gsm8k",  "gsm8k",                                  ["--limit", "200"]),
]

def free_gpus():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    return [int(l.split(",")[0]) for l in out
            if int(l.split(",")[1]) > 30000 and int(l.split(",")[2]) < 50]

def invocation_done(outdir):
    return bool(glob.glob(os.path.join(outdir, "**", "results_*.json"), recursive=True))

def run_variant(label, peft, gpu):
    """All 4 benchmark groups for one variant, sequentially, pinned to one GPU."""
    margs = f"pretrained={MODEL},{BNB}" + (f",peft={peft}" if peft else "")
    for gname, tasks, extra in GROUPS:
        outdir = os.path.join(HERE, "results_downstream", label, gname)
        if invocation_done(outdir):
            print(f"[{label}/{gname}] skip (done)", flush=True)
            continue
        os.makedirs(outdir, exist_ok=True)
        cmd = ["python3", "-m", "lm_eval", "--model", "hf", "--model_args", margs,
               "--tasks", tasks, "--batch_size", "8", "--seed", "42",
               "--output_path", outdir, "--log_samples"] + extra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME="/mnt/ssd3/hf_cache",
                   HF_DATASETS_OFFLINE="1")  # all datasets pre-cached; avoids 7-way hub rate-limit race
        log = os.path.join(HERE, f"ds_{label}_{gname}.log")
        print(f"[{label}/{gname}] start on GPU{gpu}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[{label}/{gname}] rc={rc}", flush=True)

import threading
gpus = free_gpus()
print("free GPUs:", gpus, flush=True)
assert len(gpus) >= 1, "no free GPUs"
threads = []; pool = list(gpus)
lock = threading.Lock()
queue = list(VARIANTS)

def worker():
    while True:
        with lock:
            if not queue: return
            label, peft = queue.pop(0)
            gpu = pool.pop(0)
        try:
            run_variant(label, peft, gpu)
        finally:
            with lock:
                pool.append(gpu)

for _ in range(min(len(gpus), len(VARIANTS))):
    t = threading.Thread(target=worker); t.start(); threads.append(t)
for t in threads: t.join()
print("DOWNSTREAM_SWEEP_DONE", flush=True)
