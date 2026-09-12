"""
Downstream alignment/ethics sweep for WizardLM-13B-V1.2 (SFT baseline) + per-user /
all-user DPO adapters. 4 tests scored via lm-eval 0.4.7:
  - hhh_alignment  (HHH alignment, acc)
  - sycophancy                              (model-written-evals sycophancy group, acc)
  - ethics_deontology                       (hendrycks ethics deontology subset, acc)
  - moral_stories                           (custom task, acc/acc_norm; moral vs immoral)
All 4 tasks run in ONE lm_eval call per model (amortizes the ~90s 13B-4bit load).
4-GPU pool (0,1,3,4 -- GPU2 avoided), staggered loads, resumable per model.
Output: results_downstream_wiz13b/<label>/  (one lm_eval json holding all 4 tasks).
Labels: baseline | user_<uid>_beta<b> | ALLUSERS_beta<b>  (match the OCEAN json names).
"""
import os, sys, glob, json, subprocess, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "WizardLMTeam/WizardLM-13B-V1.2"
HF_HOME = "/mnt/ssd3/tmp/goutham/hf_cache"
ADAPTER_ROOT = "/mnt/ssd3/tmp/goutham/user_adapters/WizardLM-13B-V1.2"
INCLUDE = os.path.join(HERE, "custom_tasks")
BNB = ("load_in_4bit=True,bnb_4bit_quant_type=nf4,"
       "bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True")
TASKS = "hhh_alignment,sycophancy,ethics_deontology,moral_stories"
GPUS = [0, 1, 3, 4]
OUTROOT = os.path.join(HERE, "results_downstream_wiz13b")
BETAS = ["0.01", "0.1", "0.5"]

users = json.load(open(os.path.join(HERE, "user_ids_100.json")))

# (label, adapter_dir_or_None)
VARIANTS = [("baseline", None)]
for u in users:
    for b in BETAS:
        VARIANTS.append((f"user_{u}_beta{b}", os.path.join(ADAPTER_ROOT, f"user_{u}_beta{b}")))
for b in BETAS:
    VARIANTS.append((f"ALLUSERS_beta{b}", os.path.join(ADAPTER_ROOT, f"ALL_USERS_beta{b}")))

def done(outdir):
    for p in glob.glob(os.path.join(outdir, "**", "results_*.json"), recursive=True):
        try:
            r = json.load(open(p)).get("results", {})
            keys = set(r.keys())
            # sycophancy is written as 3 subtask rows (no group aggregate row)
            has_syco = any(k.startswith("sycophancy_on") for k in keys)
            if {"hhh_alignment", "ethics_deontology", "moral_stories"} <= keys and has_syco:
                return True
        except Exception:
            pass
    return False

# build job list, logging missing adapters
missing = []
jobs = []
for label, peft in VARIANTS:
    if peft is not None and not os.path.isdir(peft):
        missing.append(label); continue
    outdir = os.path.join(OUTROOT, label)
    if done(outdir):
        continue
    jobs.append((label, peft, outdir))

os.makedirs(OUTROOT, exist_ok=True)
with open(os.path.join(HERE, "downstream_wiz13b_missing.log"), "w") as fh:
    for m in missing: fh.write(m + "\n")
print(f"{len(VARIANTS)} variants, {len(missing)} missing adapters, {len(jobs)} jobs to run", flush=True)

lock = threading.Lock()
launch_lock = threading.Lock()
queue = list(jobs); pool = list(GPUS)

def worker():
    while True:
        with lock:
            if not queue: return
            label, peft, outdir = queue.pop(0)
            gpu = pool.pop(0)
        try:
            os.makedirs(outdir, exist_ok=True)
            margs = f"pretrained={MODEL},{BNB}" + (f",peft={peft}" if peft else "")
            with launch_lock:            # stagger model loads to avoid CPU-RAM spike
                time.sleep(20)
            cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", margs,
                   "--tasks", TASKS, "--include_path", INCLUDE,
                   "--batch_size", "auto", "--seed", "42", "--limit", "500",
                   "--output_path", outdir]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=HF_HOME,
                       HF_DATASETS_TRUST_REMOTE_CODE="1", TOKENIZERS_PARALLELISM="false")
            log = os.path.join(HERE, "dswiz_logs", f"{label}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            print(f"[{label}] start GPU{gpu}", flush=True)
            t0 = time.time()
            with open(log, "w") as fh:
                rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            print(f"[{label}] rc={rc} ({time.time()-t0:.0f}s)", flush=True)
        finally:
            with lock:
                pool.append(gpu)

threads = [threading.Thread(target=worker) for _ in range(len(GPUS))]
for t in threads: t.start()
for t in threads: t.join()
print("DOWNSTREAM_WIZ13B_DONE", flush=True)
