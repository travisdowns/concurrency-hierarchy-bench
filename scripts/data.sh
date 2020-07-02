#!/bin/bash
set -euo pipefail


# echo "FAST_ITERS=${FAST_ITERS:=1000}"
# echo "SLOW_ITERS=${SLOW_ITERS:=10}"
echo "RDIR=${RDIR:=./results}"
echo "CPUS=${CPUS:=$(nproc)}"

mkdir -p "$RDIR"

# up to CPU-count threads
echo "Collecting fast data"
./bench --progress --csv > "$RDIR/data_fast.csv"
echo "Collecting slow data"
./bench --progress --csv --min-threads=$(($CPUS + 1)) --max-threads=$(($CPUS + 2)) --batch=200 --trial-time=500 > "$RDIR/data_slow.csv"

cat $RDIR/data_fast.csv <(tail +2 $RDIR/data_slow.csv) > $RDIR/combined.csv