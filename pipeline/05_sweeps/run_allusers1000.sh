#!/bin/bash
# Train + eval the 1000-user aggregate DPO for one model on one GPU.
# Usage: run_allusers1000.sh <SHORT> <HF_MODEL_ID> <GPU>
set -u
SHORT=$1; MID=$2; GPU=$3
cd /home/goutham/ocean/main
ADIR=/mnt/ssd3/user_adapters_1000/$SHORT/ALLUSERS1000_beta0.01
RES=results_${SHORT}_ALLUSERS1000_beta0.01.json
export CUDA_VISIBLE_DEVICES=$GPU
python3 -u dpo_train.py "$MID" user_dpo_all_1000.jsonl 0.01 "$ADIR" \
  && python3 -u eval_adapter.py "$MID" "$ADIR" "$RES" \
  && echo "ALLUSERS1000_DONE $SHORT -> $RES"
