#!/bin/bash
# Sleep until 08:00 local time, then take all 8 lambda GPUs for the sweep.
cd /home/goutham/ocean/main
target=$(date -d '08:00' +%s)
now=$(date +%s)
if [ "$target" -gt "$now" ]; then
  echo "sleeping $((target - now))s until 8am"
  sleep $((target - now))
fi
python3 build_lambda3_shard.py
SWEEP_GPUS=0,1,2,3,4,5,6,7 SWEEP_NGPUS=8 USERS_FILE=user_ids_1000_lambda3.json \
  ADAP_ROOT=/mnt/ssd3/user_adapters_1000 \
  nohup python3 -u master_users_1000_hermes.py > users1000_hermes_lambda3.log 2>&1 &
sleep 8
head -3 users1000_hermes_lambda3.log
echo "LAMBDA_8AM_TAKEOVER_LAUNCHED"
