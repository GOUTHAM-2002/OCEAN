"""
Re-run the under-sampled downstream tasks at FULL size for the 7 Mixtral variants
to crush the stderr bars. MMLU/HellaSwag/BoolQ/GSM8K full (ARC/Wino/TruthfulQA were
already full last run -> reused by the aggregator). 4-GPU pool, staggered, resumable.
"""
import os, glob, subprocess, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"
BNB = ("load_in_4bit=True,bnb_4bit_quant_type=nf4,"
       "bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True")
VARIANTS = [("baseline", None)] + [
    (lab, os.path.join(HERE, "adapters", lab))
    for lab in ["low_beta0.01", "low_beta0.1", "low_beta0.5",
                "high_beta0.01", "high_beta0.1", "high_beta0.5"]]
# (taskname, tasks, extra)  -- NO --limit => full dataset
TASKS = [
    ("mmlu",      "mmlu",      ["--num_fewshot", "5"]),
    ("hellaswag", "hellaswag", []),
    ("boolq",     "boolq",     []),
    ("gsm8k",     "gsm8k",     []),
]
GPUS = [0, 1, 2, 3]   # 4 concurrent Mixtral-4bit max (CPU-RAM safe with staggered loads)

def done(outdir):
    return bool(glob.glob(os.path.join(outdir, "**", "results_*.json"), recursive=True))

jobs = []
for lab, peft in VARIANTS:
    margs = f"pretrained={MODEL},{BNB}" + (f",peft={peft}" if peft else "")
    for tname, tasks, extra in TASKS:
        outdir = os.path.join(HERE, "results_downstream_full", lab, tname)
        if not done(outdir):
            jobs.append((lab, tname, margs, tasks, extra, outdir))
print(f"{len(jobs)} (variant,task) jobs to run", flush=True)

lock = threading.Lock()
launch_lock = threading.Lock()
queue = list(jobs); pool = list(GPUS)

def worker():
    while True:
        with lock:
            if not queue: return
            lab, tname, margs, tasks, extra, outdir = queue.pop(0)
            gpu = pool.pop(0)
        try:
            os.makedirs(outdir, exist_ok=True)
            with launch_lock:           # stagger model loads to avoid CPU-RAM spike
                time.sleep(60)
            cmd = ["python3", "-m", "lm_eval", "--model", "hf", "--model_args", margs,
                   "--tasks", tasks, "--batch_size", "8", "--seed", "42",
                   "--output_path", outdir] + extra
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME="/mnt/ssd3/hf_cache",
                       HF_DATASETS_OFFLINE="1", HF_DATASETS_TRUST_REMOTE_CODE="1")
            log = os.path.join(HERE, f"dsf_{lab}_{tname}.log")
            print(f"[{lab}/{tname}] start GPU{gpu}", flush=True)
            with open(log, "w") as fh:
                rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            print(f"[{lab}/{tname}] rc={rc}", flush=True)
        finally:
            with lock:
                pool.append(gpu)

threads = [threading.Thread(target=worker) for _ in range(len(GPUS))]
for t in threads: t.start()
for t in threads: t.join()
print("DOWNSTREAM_FULL_DONE", flush=True)
