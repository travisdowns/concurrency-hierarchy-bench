#!/bin/bash
# shellcheck disable=SC2162

set -euo pipefail


echo "RBASE=${RBASE:=./results}"
echo "ADIR=${ADIR:=../travisdowns.github.io/assets/concurrency-costs}"
echo "TDIR=${TDIR:=../travisdowns.github.io/misc/tables/concurrency-costs}"

uarches=(skl,4 g2-16,16 g1-16,16 icl,4)


for pair in "${uarches[@]}"; do
    IFS=',' read u _ <<< "${pair}"
    mkdir -p "$ADIR/$u"
    mkdir -p "$TDIR/$u"
done

for pair in "${uarches[@]}"; do
    IFS=',' read u procs <<< "${pair}"
    echo "uarch=${u} with $procs CPUs"
    scripts/plot-bar.py "$RBASE/${u}/combined.csv" --out "$ADIR/${u}" --table-out="$TDIR/${u}" --procs=${procs}
done