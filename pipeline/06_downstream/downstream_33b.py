"""
Downstream benchmark sweep for WizardLM-33B-Uncensored: SFT baseline + 6
AGENT-DPO adapters (low/high x beta{0.01,0.1,0.5}) on the 7-benchmark battery
via lm-eval 0.4.7. FULL size (no --limit) -- first-N subsampling was found
biased, and with 8 GPUs for 7 variants we can afford full in parallel.
One variant per GPU; resumable (skips finished groups).
Output: results_downstream_33b/<label>/<group>/...
"""
import os, glob, subprocess, threading

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "cognitivecomputations/WizardLM-33B-V1.0-Uncensored"
SHORT = "WizardLM-33B-Uncensored"
HF_HOME = "/mnt/ssd3/hf_cache"
BNB = ("load_in_4bit=True,bnb_4bit_quant_type=nf4,"
       "bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True")
ADAP = os.path.join(HERE, f"adapters_{SHORT}")
VARIANTS = [("baseline", None)] + [
    (lab, os.path.join(ADAP, lab))
    for lab in ["low_beta0.01", "low_beta0.1", "low_beta0.5",
                "high_beta0.01", "high_beta0.1", "high_beta0.5"]]
# (group_name, tasks, extra_args) -- FULL size
GROUPS = [
    ("mmlu",  "mmlu",                                    ["--num_fewshot", "5"]),
    ("hsbq",  "hellaswag,boolq",                         []),
    ("full",  "arc_challenge,truthfulqa_mc2,winogrande", []),
    ("gsm8k", "gsm8k",                                   []),
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
    margs = f"pretrained={MODEL},{BNB}" + (f",peft={peft}" if peft else "")
    for gname, tasks, extra in GROUPS:
        outdir = os.path.join(HERE, "results_downstream_33b", label, gname)
        if invocation_done(outdir):
            print(f"[{label}/{gname}] skip (done)", flush=True); continue
        os.makedirs(outdir, exist_ok=True)
        cmd = ["python3", "-m", "lm_eval", "--model", "hf", "--model_args", margs,
               "--tasks", tasks, "--batch_size", "8", "--seed", "42",
               "--output_path", outdir, "--log_samples"] + extra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=HF_HOME,
                   HF_DATASETS_OFFLINE="1")
        log = os.path.join(HERE, f"ds33_{label}_{gname}.log")
        print(f"[{label}/{gname}] start on GPU{gpu}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[{label}/{gname}] rc={rc}", flush=True)

gpus = free_gpus()
print("free GPUs:", gpus, flush=True)
assert len(gpus) >= 1, "no free GPUs"
lock = threading.Lock(); pool = list(gpus); work = list(VARIANTS)

def worker():
    while True:
        with lock:
            if not work: return
            label, peft = work.pop(0); gpu = pool.pop(0)
        try:
            run_variant(label, peft, gpu)
        finally:
            with lock: pool.append(gpu)

threads = [threading.Thread(target=worker) for _ in range(min(len(gpus), len(work)))]
for t in threads: t.start()
for t in threads: t.join()
print("DOWNSTREAM_33B_DONE", flush=True)
