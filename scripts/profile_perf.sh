#!/usr/bin/env bash
# Hardware-counter profile of the matching engine (Linux + perf only).
#
# Targets from the project spec: IPC > 2.0, L1-dcache miss rate < 1.5%.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/bench_latency
mkdir -p build
c++ -std=c++20 -O3 -DNDEBUG -march=native -g -fno-omit-frame-pointer \
    -Icpp/include cpp/bench/bench_latency.cpp -o "$BIN"

if ! command -v perf >/dev/null; then
    echo "perf not found. This script is Linux-only; on macOS use:"
    echo "  xcrun xctrace record --template 'CPU Counters' --launch -- $BIN"
    exit 1
fi

perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,\
branches,branch-misses,dTLB-load-misses "$BIN" 2000000

echo
echo "Hot-path profile:"
perf record -g --call-graph dwarf -o build/perf.data "$BIN" 1000000 >/dev/null
perf report -i build/perf.data --stdio --percent-limit 1 | head -40
