"""Downstream alignment/ethics sweep for the 1,050 World Tour DPO adapters (7 countries
x 3 age brackets x M/F) on WizardLM-13B-V1.2. Same lm-eval 0.4.7 setup as downstream_wiz13b.py.
Task battery (paper's 4 + ETHICS suite + bias):
  hhh_alignment, sycophancy(x3), ethics_cm, ethics_deontology, ethics_justice,
  ethics_utilitarianism, ethics_virtue, moral_stories, crows_pairs_english
All tasks in ONE lm_eval call per adapter. 7-GPU pool (avoid GPU6 = other user).
Output: results_downstream_worldtour/<pid>/  (pid 0..1049, plus 'baseline').
"""
import os, sys, glob, json, subprocess, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "WizardLMTeam/WizardLM-13B-V1.2"
HF_HOME = "/mnt/ssd3/hf_cache"
ADAPTER_ROOT = "/mnt/ssd3/adapters_worldtour"
INCLUDE = os.path.join(HERE, "custom_tasks")
BNB = ("load_in_4bit=True,bnb_4bit_quant_type=nf4,"
       "bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True")
TASKS = ("hhh_alignment,sycophancy,ethics_cm,ethics_deontology,ethics_justice,"
         "ethics_utilitarianism,ethics_virtue,moral_stories,crows_pairs_english")
GPUS = [0, 1, 2, 3, 4, 5, 7]
LIMIT = "500"
OUTROOT = os.path.join(HERE, "results_downstream_worldtour")
PY = sys.executable  # python3.8 with lm_eval in ~/.local

pids = sorted(int(k) for k in json.load(open(os.path.join(HERE, "worldtour_meta.json"))))
VARIANTS = [("baseline", None)] + [(str(p), os.path.join(ADAPTER_ROOT, str(p))) for p in pids]

NEED = {"hhh_alignment", "ethics_cm", "ethics_deontology", "ethics_justice",
        "ethics_utilitarianism", "ethics_virtue", "moral_stories", "crows_pairs_english"}
def done(outdir):
    for p in glob.glob(os.path.join(outdir, "**", "results_*.json"), recursive=True):
        try:
            keys = set(json.load(open(p)).get("results", {}).keys())
            if NEED <= keys and any(k.startswith("sycophancy_on") for k in keys):
                return True
        except Exception:
            pass
    return False

jobs = []
for label, peft in VARIANTS:
    if peft is not None and not os.path.isdir(peft):
        continue
    outdir = os.path.join(OUTROOT, label)
    if done(outdir):
        continue
    jobs.append((label, peft, outdir))
os.makedirs(OUTROOT, exist_ok=True)
print(f"{len(VARIANTS)} variants, {len(jobs)} jobs to run on GPUs {GPUS}", flush=True)

lock = threading.Lock(); launch_lock = threading.Lock()
queue = list(jobs)

def worker(gpu):
    while True:
        with lock:
            if not queue: return
            label, peft, outdir = queue.pop(0)
            n_left = len(queue)
        try:
            os.makedirs(outdir, exist_ok=True)
            margs = f"pretrained={MODEL},{BNB}" + (f",peft={peft}" if peft else "")
            with launch_lock:
                time.sleep(15)   # stagger loads (CPU-RAM spike)
            cmd = [PY, os.path.join(HERE, "run_lmeval_patched.py"), "--model", "hf", "--model_args", margs,
                   "--tasks", TASKS, "--include_path", INCLUDE,
                   "--batch_size", "auto", "--seed", "42", "--limit", LIMIT,
                   "--output_path", outdir]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=HF_HOME,
                       HF_DATASETS_TRUST_REMOTE_CODE="1", TOKENIZERS_PARALLELISM="false")
            log = os.path.join(HERE, "dsworld_logs", f"{label}.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            t0 = time.time()
            print(f"[{label}] start GPU{gpu} ({n_left} left)", flush=True)
            with open(log, "w") as fh:
                rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            ok = done(outdir)
            print(f"[{label}] rc={rc} ok={ok} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{label}] EXC {e}", flush=True)

threads = [threading.Thread(target=worker, args=(g,)) for g in GPUS]
for t in threads: t.start()
for t in threads: t.join()
print("WORLDTOUR_DOWNSTREAM_DONE", flush=True)
