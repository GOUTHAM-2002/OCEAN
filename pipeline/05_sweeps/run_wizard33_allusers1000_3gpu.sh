#!/bin/bash
set -euo pipefail

cd /home/goutham/ocean/main
export HF_HOME=/mnt/ssd3/hf_cache
SHORT=WizardLM-33B-Uncensored
MODEL=cognitivecomputations/WizardLM-33B-V1.0-Uncensored
ADAPTER=/mnt/ssd3/user_adapters_1000/WizardLM-33B-Uncensored/ALLUSERS1000_beta0.01

# All 1000 personalized results must be valid before we spend days on the aggregate.
python3 - <<'PY'
import json, sys
users = json.load(open("user_ids_1000.json"))
def valid(uid):
    try:
        table = json.load(open(
            f"results_WizardLM-33B-Uncensored_user_{uid}_beta0.01.json"))["table"]
        return all(
            test in table and all(
                isinstance(table[test].get(code), list) and len(table[test][code]) == 2
                for code in "OCEAN")
            for test in ("BFI-2", "FFPI"))
    except Exception:
        return False
missing = [uid for uid in users if not valid(uid)]
print(f"PRELAUNCH valid={len(users)-len(missing)}/{len(users)} missing={missing}", flush=True)
sys.exit(bool(missing))
PY

# Fresh start only: the trainer auto-resumes if checkpoints exist, and resume
# doubles the schedule (trl bug); checkpoint-7000 is also missing trainer_state.
if [[ -d "$ADAPTER" ]]; then
  rmdir "$ADAPTER" 2>/dev/null || mv "$ADAPTER" "${ADAPTER}_partial_$(date +%Y%m%d_%H%M%S)"
fi

CUDA_VISIBLE_DEVICES=1,2,5 DDP_BATCH=1 DDP_GRAD_ACCUM=3 \
python3 -m torch.distributed.run --nproc_per_node=3 --master_port=29632 \
  dpo_train_ddp.py "$MODEL" user_dpo_all_1000.jsonl 0.01 "$ADAPTER"

CUDA_VISIBLE_DEVICES=5 python3 -u eval_adapter.py "$MODEL" "$ADAPTER" \
  "results_${SHORT}_ALLUSERS1000_beta0.01.json"

SWEEP_SHORT="$SHORT" SWEEP_FAMILY=WizardLM USERS_FILE=user_ids_1000.json \
  python3 -u user_corr_plots_1000.py

echo "WIZARD33_ALLUSERS1000_PIPELINE_DONE"
