#!/bin/bash
# Babysitter for the Nous-Hermes trait DPO sweep running in tmux on mms-large-2.
# Every 10 min: heal the SSH control socket if stale, then rsync Hermes results,
# logs, and plots back here. Exits after SWEEP_DONE appears in the log.
# The sweep itself runs in tmux on mms and does NOT depend on this loop.
HOST=goutham@mms-large-2.cs.dartmouth.edu
CP=/tmp/cm_mms
SHORT=Nous-Hermes-2-Mixtral-8x7B-SFT
cd /home/goutham/ocean/main || exit 1
while true; do
  if ! ssh -o ControlPath=$CP -O check $HOST 2>/dev/null; then
    ssh -o ControlPath=$CP -O exit $HOST 2>/dev/null; rm -f $CP
    SSH_ASKPASS=/tmp/mms_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid -w ssh -fN \
      -o ControlMaster=yes -o ControlPath=$CP -o ControlPersist=72000 \
      -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new $HOST < /dev/null
  fi
  rsync -a -e "ssh -o ControlPath=$CP" \
    --include="results_${SHORT}_*.json" --include='sweep_traits_hermes.log' \
    --include="train_${SHORT}_*.log" --include="eval_${SHORT}_*.log" --exclude='*' \
    "$HOST:ocean/main/" . 2>>sync_hermes_back.err
  rsync -a -e "ssh -o ControlPath=$CP" "$HOST:ocean/main/g_plots/" g_plots/ 2>>sync_hermes_back.err
  if grep -q SWEEP_DONE sweep_traits_hermes.log 2>/dev/null; then
    echo "SYNC_LOOP_DONE $(date)" >> sync_hermes_back.err
    break
  fi
  sleep 600
done
