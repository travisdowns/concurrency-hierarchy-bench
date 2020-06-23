#!/bin/bash

# run each algorithm alone to get instruction and atomic counts
# uses --no-barrier and --warmup-ms so that the function under
# test dominates the time

set -euo pipefail

echo "EVENTS=${EVENTS:=instructions:u,mem_inst_retired.lock_loads:u}"
echo "ITERS=${ITERS=$(((33333333 + 17)/17))}"  # default should result in 100,000,000 iters

if [[ -z ${1+x} ]]; then
    algos=("mutex add" "atomic add" "cas add" "ticket yield" "ticket blocking" "queued fifo" "ticket spin" "mutex3" "cas multi" "tls add")
else
    algos=("$@")
fi

for algo in "${algos[@]}"; do
    echo "ALGO: $algo"
    perf stat -e $EVENTS ./bench --iters=$ITERS --max-threads=1 --algos="$algo" --no-barrier --warmup-ms=0 |& egrep 'instructions|lock_loads'
done


