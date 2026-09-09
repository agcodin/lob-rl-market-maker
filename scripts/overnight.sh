#!/usr/bin/env bash
# Unattended overnight training. Nothing here contacts a network service or an
# API -- it is Python and the local CPU only.
#
#   ./scripts/overnight.sh [hours]
#
# Survives the terminal closing (nohup/setsid) and keeps the Mac awake
# (caffeinate). Stop it early with: scripts/overnight.sh stop
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
DIR="$ROOT/runs/arena"
PIDFILE="$DIR/arena.pid"
LOG="$DIR/arena.log"

if [ "${1:-}" = "stop" ]; then
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        kill -TERM "$(cat "$PIDFILE")"
        echo "sent stop signal; it will finish the current generation and save"
    else
        echo "not running"
    fi
    exit 0
fi

HOURS="${1:-5.85}"
mkdir -p "$DIR"

# Pre-flight: better to find a broken test now than after six hours.
echo "pre-flight tests..."
if ! "$ROOT/.venv/bin/python" -m pytest tests -q >"$DIR/preflight.txt" 2>&1; then
    echo "tests failed -- not starting. See $DIR/preflight.txt" >&2
    tail -20 "$DIR/preflight.txt" >&2
    exit 1
fi
echo "$(tail -1 "$DIR/preflight.txt")"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running (pid $(cat "$PIDFILE")). Use '$0 stop' first." >&2
    exit 1
fi

# caffeinate -i (no idle sleep) -m (no disk sleep) -s (no system sleep on AC).
# The whole pipeline is its child, so the Mac stays awake exactly as long as
# training runs and no longer.
nohup caffeinate -ims "$ROOT/.venv/bin/python" - "$HOURS" "$ROOT" >>"$LOG" 2>&1 <<'PYEOF' &
import os, subprocess, sys
hours, root = sys.argv[1], sys.argv[2]
py = os.path.join(root, ".venv", "bin", "python")
env = dict(os.environ, PYTHONUNBUFFERED="1")
print(f"=== overnight run: {hours}h budget ===", flush=True)
rc = subprocess.call([py, "-m", "lobrl.arena", "--hours", hours,
                      "--n-agents", "4", "--learner-seats", "2",
                      "--chunk-transitions", "400000", "--eval-episodes", "32",
                      "--episode-steps", "1000", "--out", "runs/arena"],
                     cwd=root, env=env)
print(f"=== training exited rc={rc}; running tests ===", flush=True)
subprocess.call([py, "-m", "pytest", "tests", "-q"], cwd=root, env=env)
print("=== building report ===", flush=True)
subprocess.call([py, "scripts/arena_report.py", "--dir", "runs/arena"], cwd=root, env=env)
for cmd in (["scripts/record_league.py", "--sizes", "1", "2", "4", "8", "--episodes", "12"],
            ["scripts/build_dashboard.py"]):
    print(f"=== {' '.join(cmd)} ===", flush=True)
    subprocess.call([py] + cmd, cwd=root, env=env)
print("=== ALL DONE ===", flush=True)
PYEOF

echo $! > "$PIDFILE"
echo "started (pid $(cat "$PIDFILE")), ${HOURS}h budget"
echo "  progress : cat runs/arena/status.json"
echo "  log      : tail -f runs/arena/arena.log"
echo "  report   : runs/arena/MORNING_REPORT.md (written when it finishes)"
echo "  stop     : $0 stop"
