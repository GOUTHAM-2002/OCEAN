#!/bin/bash
set -euo pipefail

cd /home/goutham/ocean/main
SHORT=WizardLM-33B-Uncensored
MODEL=cognitivecomputations/WizardLM-33B-V1.0-Uncensored
ADAPTER=/mnt/ssd3/user_adapters_1000/WizardLM-33B-Uncensored/ALLUSERS1000_beta0.01

# Wait for the two-GPU retry driver to finish.
while pgrep -f '^python3 -u master_users_1000_hermes.py$' >/dev/null; do
  sleep 30
done

# Do not replace the aggregate run unless all personalized results are valid.
python3 - <<'PY'
import json
import sys

users = json.load(open("user_ids_1000.json"))

def valid(uid):
    try:
        table = json.load(open(
            f"results_WizardLM-33B-Uncensored_user_{uid}_beta0.01.json"
        ))["table"]
        return all(
            test in table and all(
                isinstance(table[test].get(code), list)
                and len(table[test][code]) == 2
                for code in "OCEAN"
            )
            for test in ("BFI-2", "FFPI")
        )
    except Exception:
        return False

missing = [uid for uid in users if not valid(uid)]
print(f"POST_RETRY valid={len(users) - len(missing)}/{len(users)} missing={missing}", flush=True)
sys.exit(bool(missing))
PY

# The legacy single-GPU trainer cannot resume under DDP. Stop it only after
# retries pass, then remove exactly its partial artifact before a fresh DDP run.
old_pid=$(pgrep -f '^python3 -u dpo_train.py cognitivecomputations/WizardLM-33B-V1.0-Uncensored user_dpo_all_1000.jsonl 0.01 /mnt/ssd3/user_adapters_1000/WizardLM-33B-Uncensored/ALLUSERS1000_beta0.01$' || true)
if [[ -n "$old_pid" ]]; then
  kill -TERM "$old_pid"
  while kill -0 "$old_pid" 2>/dev/null; do sleep 2; done
fi

if [[ "$ADAPTER" != "/mnt/ssd3/user_adapters_1000/WizardLM-33B-Uncensored/ALLUSERS1000_beta0.01" ]]; then
  echo "Refusing unexpected cleanup target: $ADAPTER" >&2
  exit 20
fi
if [[ -d "$ADAPTER" ]]; then
  find "$ADAPTER" -depth -mindepth 1 -delete
  rmdir "$ADAPTER"
fi

CUDA_VISIBLE_DEVICES=0,1,2 DDP_BATCH=1 DDP_GRAD_ACCUM=3 \
python3 -m torch.distributed.run --nproc_per_node=3 --master_port=29631 \
  dpo_train_ddp.py "$MODEL" user_dpo_all_1000.jsonl 0.01 "$ADAPTER"

CUDA_VISIBLE_DEVICES=0 python3 -u eval_adapter.py "$MODEL" "$ADAPTER" \
  "results_${SHORT}_ALLUSERS1000_beta0.01.json"

SWEEP_SHORT="$SHORT" SWEEP_FAMILY=WizardLM USERS_FILE=user_ids_1000.json \
  python3 -u user_corr_plots_1000.py
