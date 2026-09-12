"""
OCEAN x behavioral-evals: run the 14-eval battery on the 7 Mixtral variants
(SFT baseline + 6 personality-DPO adapters). GPUs 3-7 only (0-2 reserved).
Resumable; identical docs across variants (first-N limits, seed 42).
"""
import os, glob, subprocess, time, threading

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"
BNB = ("load_in_4bit=True,bnb_4bit_quant_type=nf4,"
       "bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True")
VARIANTS = [("baseline", None)] + [
    (lab, os.path.join(HERE, "adapters", lab))
    for lab in ["low_beta0.01", "low_beta0.1", "low_beta0.5",
                "high_beta0.01", "high_beta0.1", "high_beta0.5"]]
BEH = ("sycophancy,toxigen,crows_pairs_english,ethics_cm,ethics_justice,"
       "advanced_ai_risk_human-power-seeking-inclination,"
       "advanced_ai_risk_human-survival-instinct,"
       "advanced_ai_risk_human-corrigible-less-HHH,"
       "persona_machiavellianism,persona_narcissism,persona_psychopathy")
GROUPS = [
    ("beh",  BEH,                        ["--limit", "200"]),
    ("tqhs", "truthfulqa_mc2,hellaswag", ["--limit", "400"]),
    ("mmlu", "mmlu",                     ["--num_fewshot", "5", "--limit", "10"]),
]
GPUS = [3, 4, 5, 6, 7]

def invocation_done(outdir):
    return bool(glob.glob(os.path.join(outdir, "**", "results_*.json"), recursive=True))

def run_variant(label, peft, gpu):
    margs = f"pretrained={MODEL},{BNB}" + (f",peft={peft}" if peft else "")
    for gname, tasks, extra in GROUPS:
        outdir = os.path.join(HERE, "results_oceancorr", label, gname)
        if invocation_done(outdir):
            print(f"[{label}/{gname}] skip", flush=True); continue
        os.makedirs(outdir, exist_ok=True)
        cmd = ["python3", "-m", "lm_eval", "--model", "hf", "--model_args", margs,
               "--tasks", tasks, "--batch_size", "8", "--seed", "42",
               "--output_path", outdir] + extra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME="/mnt/ssd3/hf_cache",
                   HF_DATASETS_OFFLINE="1", HF_DATASETS_TRUST_REMOTE_CODE="1")
        log = os.path.join(HERE, f"oc_{label}_{gname}.log")
        print(f"[{label}/{gname}] start GPU{gpu}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[{label}/{gname}] rc={rc}", flush=True)

lock = threading.Lock(); queue = list(VARIANTS); pool = list(GPUS)
def worker():
    while True:
        with lock:
            if not queue: return
            label, peft = queue.pop(0); gpu = pool.pop(0)
        try: run_variant(label, peft, gpu)
        finally:
            with lock: pool.append(gpu)

threads = [threading.Thread(target=worker) for _ in range(min(len(GPUS), len(VARIANTS)))]
for t in threads: t.start()
for t in threads: t.join()
print("OCEANCORR_DONE", flush=True)
