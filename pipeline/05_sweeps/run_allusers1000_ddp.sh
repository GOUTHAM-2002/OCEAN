#!/bin/bash
# Train (DDP, 4 GPUs) + eval (1 GPU) the 1000-user aggregate DPO for one model.
# Usage: run_allusers1000_ddp.sh <SHORT> <HF_MODEL_ID> <GPU_CSV_4> <MASTER_PORT>
set -u
SHORT=$1; MID=$2; GPUS=$3; PORT=$4
cd /home/goutham/ocean/main
ADIR=/mnt/ssd3/user_adapters_1000/$SHORT/ALLUSERS1000_beta0.01
RES=results_${SHORT}_ALLUSERS1000_beta0.01.json
export CUDA_VISIBLE_DEVICES=$GPUS
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
# keep effective batch = 8: ranks x batch2 x accum = 8. NOTE: resume from a
# checkpoint is BROKEN in this trl/transformers combo (doubles the schedule
# to 18750 steps) — aggregates must always train fresh; verify the displayed
# total is 9375 after launch.
export DDP_BATCH=2
export DDP_GRAD_ACCUM=$((8 / NP / 2))
python3 -m torch.distributed.run --nproc_per_node=$NP --master_port=$PORT dpo_train_ddp.py \
  "$MID" user_dpo_all_1000.jsonl 0.01 "$ADIR" \
  && CUDA_VISIBLE_DEVICES=${GPUS%%,*} python3 -u eval_adapter.py "$MID" "$ADIR" "$RES" \
  && echo "ALLUSERS1000_DONE $SHORT -> $RES"
