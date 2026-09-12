#!/bin/bash
# Wait for the lambda driver to finish its own shard, then start helping mms.
cd /home/goutham/ocean/main
while pgrep -f 'master_users_1000_hermes[.]py' >/dev/null; do
  sleep 120
done
echo "lambda driver finished — starting mms helper"
nohup python3 -u helper_mms_tail.py > helper_mms_tail.log 2>&1 &
sleep 5
head -3 helper_mms_tail.log
echo "HELPER_LAUNCHED"
