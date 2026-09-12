#!/bin/bash
# Babysitter for the single-trait DPO sweep running in tmux on mms-large-2.
# Every 10 min: heal the SSH control socket if stale, then rsync trait results,
# logs, and plots back to lambda. Exits after SWEEP_DONE appears in the log.
# The sweep itself runs in tmux on mms and does NOT depend on this loop.
HOST=goutham@mms-large-2.cs.dartmouth.edu
CP=/tmp/cm_mms
cd /home/goutham/ocean/main || exit 1
while true; do
  if ! ssh -o ControlPath=$CP -O check $HOST 2>/dev/null; then
    ssh -o ControlPath=$CP -O exit $HOST 2>/dev/null; rm -f $CP
    SSH_ASKPASS=/tmp/mms_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 setsid -w ssh -fN \
      -o ControlMaster=yes -o ControlPath=$CP -o ControlPersist=72000 \
      -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new $HOST < /dev/null
  fi
  rsync -a -e "ssh -o ControlPath=$CP" \
    --include='results_*trait*.json' --include='sweep_traits.log' \
    --include='train_*trait*.log' --include='eval_*trait*.log' --exclude='*' \
    "$HOST:ocean/main/" . 2>>sync_traits_back.err
  rsync -a -e "ssh -o ControlPath=$CP" "$HOST:ocean/main/g_plots/" g_plots/ 2>>sync_traits_back.err
  if grep -q SWEEP_DONE sweep_traits.log 2>/dev/null; then
    echo "SYNC_LOOP_DONE $(date)" >> sync_traits_back.err
    break
  fi
  sleep 600
done
