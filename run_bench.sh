#!/bin/bash
# run_bench.sh PATH STEPS SEQ CAP: one guarded benchmark run, log + json in results/
cd "$(dirname "$0")"; mkdir -p results
L=results/bench_$1_seq$3.log
~/.local/mlx-server/bin/python bench_train.py --path $1 --steps $2 --seq $3 --out results/bench_$1_seq$3.json > $L 2>&1 &
P=$!; ./guard.sh $P $L ${4:-600} & wait $P; echo "exit $? " >> $L
