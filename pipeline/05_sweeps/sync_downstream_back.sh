#!/bin/bash
# Babysitter for the single-trait downstream benchmark sweep (tmux "dssweep" on mms).
# Every 10 min: heal the SSH control socket if stale, then rsync the small
# results_*.json files (NOT the big log_samples jsonl) and logs back here.
# Exits after DOWNSTREAM_TRAITS_DONE appears in the sweep log.
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
  # only the aggregate metric jsons + dir structure (skip samples_*.jsonl)
  rsync -a -m -e "ssh -o ControlPath=$CP" \
    --include='*/' --include='results_*.json' --exclude='*' \
    "$HOST:ocean/main/results_downstream_traits/" results_downstream_traits/ 2>>sync_downstream_back.err
  rsync -a -e "ssh -o ControlPath=$CP" \
    --include='dst_*.log' --include='sweep_downstream_traits.log' \
    --include='stats_downstream_traits.log' --exclude='*' \
    "$HOST:ocean/main/" . 2>>sync_downstream_back.err
  # pull the stats tables + plots once the finalizer writes them
  rsync -a -e "ssh -o ControlPath=$CP" "$HOST:ocean/main/g_plots/" g_plots/ 2>>sync_downstream_back.err
  if grep -q DOWNSTREAM_STATS_DONE sweep_downstream_traits.log 2>/dev/null; then
    echo "SYNC_LOOP_DONE $(date)" >> sync_downstream_back.err
    break
  fi
  sleep 600
done
