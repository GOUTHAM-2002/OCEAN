"""Phase C driver v2: one per_variant_eval.py per GPU, STAGGERED launches so the
adapter embedding-resizes don't contend on CPU. Each variant loads the model once
and runs all 4 groups (batch_size=auto). Resumable; retries a dead variant once.
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = "WizardLM-33B-Uncensored"
ADAP = os.path.join(HERE, f"adapters_{SHORT}")
VARIANTS = [
    ("baseline", "none"),
    ("low_beta0.01", os.path.join(ADAP, "low_beta0.01")),
    ("low_beta0.1",  os.path.join(ADAP, "low_beta0.1")),
    ("low_beta0.5",  os.path.join(ADAP, "low_beta0.5")),
    ("high_beta0.01", os.path.join(ADAP, "high_beta0.01")),
    ("high_beta0.1",  os.path.join(ADAP, "high_beta0.1")),
    ("high_beta0.5",  os.path.join(ADAP, "high_beta0.5")),
]
GPUS = [0, 1, 2, 3, 4, 5, 6]   # one variant per GPU; GPU7 spare
STAGGER = 150                  # seconds between launches (spreads the resizes)

def launch(variant, peft, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = os.path.join(HERE, f"pv_{variant}.log")
    p = subprocess.Popen(["python3", "-u", "per_variant_eval.py", variant, peft],
                         env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    print(f"  launch {variant} -> GPU{gpu} pid {p.pid}", flush=True)
    return p

running = {}   # variant -> (popen, gpu, attempt)
for i, (variant, peft) in enumerate(VARIANTS):
    gpu = GPUS[i]
    running[variant] = (launch(variant, peft, gpu), gpu, 1)
    if i < len(VARIANTS) - 1:
        time.sleep(STAGGER)

print("all launched; monitoring", flush=True)
while running:
    time.sleep(30)
    for variant in list(running):
        p, gpu, attempt = running[variant]
        if p.poll() is None:
            continue
        rc = p.returncode
        print(f"  {variant} exited rc={rc} (attempt {attempt})", flush=True)
        # check if it actually finished all groups
        peft = dict(VARIANTS)[variant]
        import glob
        groups = ["mmlu", "hsbq", "full", "gsm8k"]
        missing = [g for g in groups
                   if not glob.glob(os.path.join(HERE, "results_downstream_33b", variant, g, "results_*.json"))]
        if missing and attempt < 2:
            print(f"  {variant} missing {missing}; retry", flush=True)
            running[variant] = (launch(variant, peft, gpu), gpu, attempt + 1)
        else:
            if missing:
                print(f"  {variant} GAVE UP, still missing {missing}", flush=True)
            del running[variant]

print("DOWNSTREAM_33B_V2_DONE", flush=True)
