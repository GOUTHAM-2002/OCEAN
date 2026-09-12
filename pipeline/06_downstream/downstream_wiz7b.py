"""
Downstream alignment/ethics sweep for wizardLM-7B (TheBloke/wizardLM-7B-HF, SFT
baseline) + per-user / all-user DPO adapters. Runs on lambda. 4 tests via lm-eval:
  - hhh_alignment  (HHH alignment, acc)
  - sycophancy                              (model-written-evals sycophancy group, acc)
  - ethics_deontology                       (hendrycks ethics deontology subset, acc)
  - moral_stories    (custom task, acc/acc_norm; moral vs immoral)
All 4 tasks run in ONE lm_eval call per model. Resumable per model.
GPU pool: auto-detected idle GPUs at runtime (refreshed as GPUs free up), same
free-GPU approach as the training scripts -- uses whatever is available.
Output: results_downstream_wiz7b/<label>/  (one lm_eval json holding all 4 tasks).
Labels: baseline | user_<uid>_beta<b> | ALLUSERS_beta<b>  (match the OCEAN json names).
"""
import os, sys, glob, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "TheBloke/wizardLM-7B-HF"
HF_HOME = "/mnt/ssd3/hf_cache"
ADAPTER_ROOT = "/mnt/ssd3/user_adapters/wizardLM-7B"
INCLUDE = os.path.join(HERE, "custom_tasks")
BNB = ("load_in_4bit=True,bnb_4bit_quant_type=nf4,"
       "bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True")
TASKS = "hhh_alignment,sycophancy,ethics_deontology,moral_stories"
OUTROOT = os.path.join(HERE, "results_downstream_wiz7b")
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
with open(os.path.join(HERE, "downstream_wiz7b_missing.log"), "w") as fh:
    for m in missing: fh.write(m + "\n")
print(f"{len(VARIANTS)} variants, {len(missing)} missing adapters, {len(jobs)} jobs to run", flush=True)

def free_gpus():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    return [int(l.split(",")[0]) for l in out
            if int(l.split(",")[1]) > 20000 and int(l.split(",")[2]) < 50]

def launch(label, peft, outdir, gpu):
    os.makedirs(outdir, exist_ok=True)
    margs = f"pretrained={MODEL},{BNB}" + (f",peft={peft}" if peft else "")
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", margs,
           "--tasks", TASKS, "--include_path", INCLUDE,
           "--batch_size", "auto", "--seed", "42", "--limit", "500",
           "--output_path", outdir]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=HF_HOME,
               HF_DATASETS_TRUST_REMOTE_CODE="1", TOKENIZERS_PARALLELISM="false")
    logdir = os.path.join(HERE, "dswiz7b_logs"); os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, f"{label}.log")
    fh = open(log, "w")
    print(f"[{label}] start GPU{gpu}", flush=True)
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh, gpu, label

# dynamic scheduler: keep every idle GPU busy, refresh as they free up
q = list(jobs)
gpus = free_gpus()
while not gpus and q:
    time.sleep(30); gpus = free_gpus()
pool = list(gpus); running = {}
while q or running:
    gpus = free_gpus()
    for g in gpus:
        if g not in pool and g not in [v[2] for v in running.values()]:
            pool.append(g)
    while q and pool:
        label, peft, outdir = q.pop(0); g = pool.pop(0)
        p, fh, gpu, lab = launch(label, peft, outdir, g)
        running[p] = (fh, lab, gpu)
        time.sleep(20)   # stagger model loads to avoid CPU-RAM spikes
    time.sleep(10)
    for p in list(running):
        if p.poll() is not None:
            fh, lab, gpu = running.pop(p); fh.close()
            print(f"[{lab}] rc={p.returncode}", flush=True)
            if gpu not in pool: pool.append(gpu)
print("DOWNSTREAM_WIZ7B_DONE", flush=True)
