#!/bin/bash
# Self-healing, fully-detached daemon for the 20-user sweep.
# Keeps the (resumable) master alive until USERS_SWEEP_DONE. Survives SSH/terminal close.
cd /home/goutham/ocean/main
echo "[daemon $(date)] start (pid $$, sid $(ps -o sid= -p $$))" >> daemon.log
while ! grep -q "USERS_SWEEP_DONE" master_users.log 2>/dev/null; do
  # clear any stray workers from a previous master so we never double-train the same adapter dir
  pkill -f "python3 -u dpo_train.py" 2>/dev/null
  pkill -f "python3 -u eval_adapter.py" 2>/dev/null
  sleep 4
  echo "[daemon $(date)] launching master" >> daemon.log
  python3 -u master_users.py >> master_users.log 2>&1
  echo "[daemon $(date)] master exited rc=$? ; will relaunch unless done" >> daemon.log
  sleep 5
done
echo "[daemon $(date)] USERS_SWEEP_DONE -- daemon exiting" >> daemon.log
