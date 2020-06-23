#!/bin/bash
set -euo pipefail


echo "FAST_ITERS=${FAST_ITERS:=1000}"
echo "SLOW_ITERS=${SLOW_ITERS:=10}"
echo "RDIR=${RDIR:=./results}"
echo "CPUS=${CPUS:=$(nproc)}"

mkdir -p "$RDIR"

# up to CPU-count threads
echo "Collecting fast data"
./bench --csv --iters=$FAST_ITERS > "$RDIR/data_fast.csv"
echo "Collecting slow data"
./bench --csv --iters=$SLOW_ITERS --min-threads=$(($CPUS + 1)) --max-threads=$(($CPUS + 2)) --csv > "$RDIR/data_slow.csv"

cat $RDIR/data_fast.csv <(tail +2 $RDIR/data_slow.csv) > $RDIR/combined.csv