"""
Downstream benchmark sweep for single-trait "control" DPO adapters.

For all 7 models: SFT baseline (no adapter) + 30 single-trait adapters
(trait{O,C,E,A,N}_{high,low}_beta{0.01,0.1,0.5}) across the same 7-benchmark
battery / 4 lm-eval groups used in the earlier Hermes downstream run, so the
numbers stay comparable (lm-eval 0.4.7, 4-bit nf4, batch 8, seed 42).

Scheduler: one (model,variant) per GPU, all 4 groups sequential within a
variant; variants run in parallel across the free GPUs. Resumable: skips any
(variant,group) whose results_*.json already exists.

Output layout: results_downstream_traits/<short>/<variant>/<group>/...

Env knobs (for the smoke test / partial runs):
  DS_ONLY_MODEL=<short>            restrict to a single model short name
  DS_ONLY_VARIANTS=v1,v2,...       restrict to a comma list of variant labels
Set neither for the full 7-model x 31-variant matrix.
"""
import os, glob, subprocess, threading, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
HF_HOME = "/mnt/ssd3/tmp/goutham/hf_cache"

BNB = ("load_in_4bit=True,bnb_4bit_quant_type=nf4,"
       "bnb_4bit_compute_dtype=bfloat16,bnb_4bit_use_double_quant=True")

# (family, short, hf_id)
MODELS = [
    ("Tulu-2",      "tulu-2-7b",                       "allenai/tulu-2-7b"),
    ("Tulu-2",      "tulu-2-13b",                      "allenai/tulu-2-13b"),
    ("Vicuna",      "vicuna-7b-v1.5",                  "lmsys/vicuna-7b-v1.5"),
    ("Vicuna",      "vicuna-13b-v1.5",                 "lmsys/vicuna-13b-v1.5"),
    ("WizardLM",    "wizardLM-7B",                     "TheBloke/wizardLM-7B-HF"),
    ("WizardLM",    "WizardLM-13B-V1.2",               "WizardLMTeam/WizardLM-13B-V1.2"),
    ("Nous-Hermes", "Nous-Hermes-2-Mixtral-8x7B-SFT",  "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"),
]

TRAITS = ["O", "C", "E", "A", "N"]
DIRS   = ["high", "low"]
BETAS  = ["0.01", "0.1", "0.5"]

def trait_variants():
    return [f"trait{T}_{d}_beta{b}" for T in TRAITS for d in DIRS for b in BETAS]

# (group_name, tasks, extra_args) -- identical to downstream_eval.py
GROUPS = [
    ("mmlu",  "mmlu",                                    ["--num_fewshot", "5", "--limit", "25"]),
    ("hsbq",  "hellaswag,boolq",                         ["--limit", "1000"]),
    ("full",  "arc_challenge,truthfulqa_mc2,winogrande", []),
    ("gsm8k", "gsm8k",                                   ["--limit", "200"]),
]

PY = "/mnt/ssd3/tmp/goutham/venv_dpo/bin/python"


def free_gpus():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"]).decode().strip().splitlines()
    return [int(l.split(",")[0]) for l in out
            if int(l.split(",")[1].strip()) > 35000 and int(l.split(",")[2].strip()) < 50]


def invocation_done(outdir):
    return bool(glob.glob(os.path.join(outdir, "**", "results_*.json"), recursive=True))


def adapter_path(short, variant):
    return os.path.join(HERE, f"adapters_{short}", variant)


def maybe_seed_hermes_baseline(short, hf_id):
    """Reuse the earlier Hermes baseline downstream numbers if present, copying
    them into the new layout so we don't recompute Mixtral baseline."""
    if short != "Nous-Hermes-2-Mixtral-8x7B-SFT":
        return
    safe = hf_id.replace("/", "__")
    for gname, _, _ in GROUPS:
        dst = os.path.join(HERE, "results_downstream_traits", short, "baseline", gname)
        if invocation_done(dst):
            continue
        src = os.path.join(HERE, "results_downstream", "baseline", gname, safe)
        srcfiles = glob.glob(os.path.join(src, "results_*.json"))
        if not srcfiles:
            continue
        os.makedirs(os.path.join(dst, safe), exist_ok=True)
        for f in glob.glob(os.path.join(src, "*")):
            shutil.copy2(f, os.path.join(dst, safe, os.path.basename(f)))
        print(f"[{short}/baseline/{gname}] seeded from results_downstream/", flush=True)


def run_variant(family, short, hf_id, variant, gpu):
    peft = None if variant == "baseline" else adapter_path(short, variant)
    margs = f"pretrained={hf_id},{BNB}" + (f",peft={peft}" if peft else "")
    for gname, tasks, extra in GROUPS:
        outdir = os.path.join(HERE, "results_downstream_traits", short, variant, gname)
        if invocation_done(outdir):
            print(f"[{short}/{variant}/{gname}] skip (done)", flush=True)
            continue
        os.makedirs(outdir, exist_ok=True)
        cmd = [PY, "-m", "lm_eval", "--model", "hf", "--model_args", margs,
               "--tasks", tasks, "--batch_size", "8", "--seed", "42",
               "--output_path", outdir, "--log_samples"] + extra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=HF_HOME,
                   HF_DATASETS_OFFLINE="1")
        log = os.path.join(HERE, f"dst_{short}_{variant}_{gname}.log")
        print(f"[{short}/{variant}/{gname}] start on GPU{gpu}", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        print(f"[{short}/{variant}/{gname}] rc={rc}", flush=True)


def build_queue():
    only_model = os.environ.get("DS_ONLY_MODEL", "").strip()
    only_vars = [v for v in os.environ.get("DS_ONLY_VARIANTS", "").split(",") if v.strip()]
    q = []
    for family, short, hf_id in MODELS:
        if only_model and short != only_model:
            continue
        maybe_seed_hermes_baseline(short, hf_id)
        variants = ["baseline"] + trait_variants()
        if only_vars:
            variants = [v for v in variants if v in only_vars]
        for v in variants:
            q.append((family, short, hf_id, v))
    return q


def main():
    queue = build_queue()
    print(f"queued {len(queue)} (model,variant) jobs", flush=True)
    gpus = free_gpus()
    print("free GPUs:", gpus, flush=True)
    assert len(gpus) >= 1, "no free GPUs"

    lock = threading.Lock()
    pool = list(gpus)
    work = list(queue)

    def worker():
        while True:
            with lock:
                if not work:
                    return
                family, short, hf_id, variant = work.pop(0)
                gpu = pool.pop(0)
            try:
                run_variant(family, short, hf_id, variant, gpu)
            finally:
                with lock:
                    pool.append(gpu)

    threads = [threading.Thread(target=worker) for _ in range(min(len(gpus), len(work)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("DOWNSTREAM_TRAITS_DONE", flush=True)


if __name__ == "__main__":
    main()
