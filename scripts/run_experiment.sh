#!/usr/bin/env bash
# Train one experiment arm in the background. Usage: run_experiment.sh <dir> [args...]
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="runs/$1"; shift
mkdir -p "$DIR"
nohup caffeinate -ims .venv/bin/python -m lobrl.arena \
    --n-agents 4 --learner-seats 2 --chunk-transitions 150000 \
    --eval-episodes 32 --episode-steps 1000 --out "$DIR" "$@" \
    >> "$DIR/run.log" 2>&1 &
echo $! > "$DIR/pid"
echo "started $DIR (pid $(cat "$DIR/pid"))"
