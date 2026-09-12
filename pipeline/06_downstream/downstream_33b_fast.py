"""
Phase C (robust + fast): WizardLM-33B-Uncensored downstream on SFT baseline + 6
agent adapters, 7-benchmark battery, FULL size. Parallelized at the
(variant, group) level across ALL GPUs, batch_size=auto (no OOM), per-job
watchdog (timeout + one retry). Resumable: skips any (variant,group) already
having results_*.json. Output: results_downstream_33b/<variant>/<group>/.
"""
import os, glob, subprocess, threading, queue, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
# (group, tasks, extra_args)  -- batch_size=auto handles OOM for all
GROUPS = [
    ("mmlu",  "mmlu",                                    ["--num_fewshot", "5"]),
    ("hsbq",  "hellaswag,boolq",                         []),
    ("full",  "arc_challenge,truthfulqa_mc2,winogrande", []),
    ("gsm8k", "gsm8k",                                   []),
]
GPUS = list(range(8))
JOB_TIMEOUT = 5400   # 90 min/job hard cap; watchdog kills + retries once

def invocation_done(outdir):
    return bool(glob.glob(os.path.join(outdir, "**", "results_*.json"), recursive=True))

def build_jobs():
    jobs = []
    # mmlu first (longest pole), then the rest
    order = {"mmlu": 0, "hsbq": 1, "gsm8k": 2, "full": 3}
    for (lab, peft) in VARIANTS:
        for (g, tasks, extra) in GROUPS:
            outdir = os.path.join(HERE, "results_downstream_33b", lab, g)
            if invocation_done(outdir):
                continue
            jobs.append((lab, peft, g, tasks, extra, outdir))
    jobs.sort(key=lambda j: order.get(j[2], 9))
    return jobs

def run_one(job, gpu, attempt):
    lab, peft, g, tasks, extra, outdir = job
    os.makedirs(outdir, exist_ok=True)
    margs = f"pretrained={MODEL},{BNB}" + (f",peft={peft}" if peft else "")
    cmd = ["python3", "-m", "lm_eval", "--model", "hf", "--model_args", margs,
           "--tasks", tasks, "--batch_size", "auto", "--seed", "42",
           "--output_path", outdir, "--log_samples"] + extra
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=HF_HOME,
               HF_DATASETS_OFFLINE="1")
    log = os.path.join(HERE, f"ds33f_{lab}_{g}.log")
    print(f"[{lab}/{g}] start GPU{gpu} attempt{attempt}", flush=True)
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=JOB_TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
            print(f"[{lab}/{g}] TIMEOUT on GPU{gpu}", flush=True)
            return False
    ok = (rc == 0) and invocation_done(outdir)
    print(f"[{lab}/{g}] rc={rc} ok={ok} GPU{gpu}", flush=True)
    return ok

Q = queue.Queue()
for j in build_jobs():
    Q.put((j, 1))   # (job, attempt)
print(f"queued {Q.qsize()} (variant,group) jobs across {len(GPUS)} GPUs", flush=True)

lock = threading.Lock()
fails = []
def worker(gpu):
    while True:
        try:
            job, attempt = Q.get_nowait()
        except queue.Empty:
            return
        try:
            ok = run_one(job, gpu, attempt)
        except Exception as e:
            print(f"[{job[0]}/{job[2]}] EXC {e}", flush=True); ok = False
        if not ok:
            if attempt < 2:
                Q.put((job, attempt + 1))     # retry once
                print(f"[{job[0]}/{job[2]}] requeued (attempt {attempt+1})", flush=True)
            else:
                with lock: fails.append((job[0], job[2]))
                print(f"[{job[0]}/{job[2]}] GAVE UP after 2 attempts", flush=True)
        Q.task_done()

threads = [threading.Thread(target=worker, args=(g,)) for g in GPUS]
for t in threads: t.start()
for t in threads: t.join()
print(f"DOWNSTREAM_33B_FAST_DONE | failures: {fails}", flush=True)
