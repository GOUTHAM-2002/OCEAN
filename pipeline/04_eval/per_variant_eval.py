"""Evaluate ONE variant of WizardLM-33B on the 7-benchmark battery, loading the
model (and adapter) exactly ONCE and running all 4 groups in-process. This avoids
the per-group model reload + slow adapter embedding-resize that made the parallel
(variant,group) runner crawl. batch_size=auto (no OOM). Resumable per group.
Usage: python per_variant_eval.py <variant_label> <peft_path|none>
"""
import os, sys, json, glob
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HOME", "/mnt/ssd3/hf_cache")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "cognitivecomputations/WizardLM-33B-V1.0-Uncensored"
VARIANT, PEFT = sys.argv[1], sys.argv[2]
# (group, tasks, num_fewshot, limit)  -- limit is per-subtask; fixed seed => identical
# docs across all 7 variants, so the baseline-vs-adapter comparison is unbiased.
GROUPS = [
    ("mmlu",  ["mmlu"],                                          5,    25),
    ("hsbq",  ["hellaswag", "boolq"],                            None, 1000),
    ("full",  ["arc_challenge", "truthfulqa_mc2", "winogrande"], None, None),
    ("gsm8k", ["gsm8k"],                                         None, 100),
]

def done(group):
    d = os.path.join(HERE, "results_downstream_33b", VARIANT, group)
    return bool(glob.glob(os.path.join(d, "results_*.json")))

todo = [g for g in GROUPS if not done(g[0])]
print(f"[{VARIANT}] groups to run: {[g[0] for g in todo]}", flush=True)
# fast loglikelihood groups first (early results); slow generative gsm8k LAST
todo.sort(key=lambda g: {"full": 0, "hsbq": 1, "mmlu": 2, "gsm8k": 3}[g[0]])
if not todo:
    print(f"[{VARIANT}] all groups done", flush=True); sys.exit(0)

# fixed conservative batch (reused model + auto-batch fragments memory across
# groups -> OOM; a fixed batch + empty_cache between groups is robust).
PER_GROUP_BATCH = {"full": 8, "hsbq": 8, "mmlu": 4, "gsm8k": 4}
margs = dict(pretrained=MODEL, load_in_4bit=True, bnb_4bit_quant_type="nf4",
             bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True,
             batch_size=4, trust_remote_code=False)
if PEFT != "none":
    margs["peft"] = PEFT
print(f"[{VARIANT}] loading model (once)...", flush=True)
lm = HFLM(**margs)
print(f"[{VARIANT}] model ready", flush=True)

for group, tasks, nf, limit in todo:
    outdir = os.path.join(HERE, "results_downstream_33b", VARIANT, group)
    os.makedirs(outdir, exist_ok=True)
    torch.cuda.empty_cache()
    lm.batch_size_per_gpu = PER_GROUP_BATCH.get(group, 4)   # per-group fixed batch
    print(f"[{VARIANT}/{group}] start (tasks={tasks} nf={nf} limit={limit} bs={lm.batch_size_per_gpu})", flush=True)
    try:
        res = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=nf, limit=limit,
                                      random_seed=42, numpy_random_seed=42, torch_random_seed=42)
        out = {"results": res["results"], "configs": res.get("configs", {}),
               "variant": VARIANT, "group": group}
        with open(os.path.join(outdir, f"results_{group}.json"), "w") as f:
            json.dump(out, f, default=str, indent=2)
        accs = {t: {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}
                for t, m in res["results"].items()}
        print(f"[{VARIANT}/{group}] DONE {accs}", flush=True)
    except Exception as e:
        print(f"[{VARIANT}/{group}] FAILED: {type(e).__name__}: {e}", flush=True)

print(f"[{VARIANT}] VARIANT_DONE", flush=True)
