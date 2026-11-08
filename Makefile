PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install bench bench-py test cpp-test cpp-test-san train eval selfplay sweep league dashboard overnight overnight-stop arena-status perf clean

venv:
	python3 -m venv .venv && $(PIP) install -U pip wheel setuptools

install: venv
	$(PIP) install -r requirements.txt
	ARCHFLAGS="-arch $$(uname -m)" LOBRL_NATIVE=1 $(PIP) install -e .

test: cpp-test
	$(PY) -m pytest tests -q

# --- C++ engine -----------------------------------------------------------
build/CMakeCache.txt:
	cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

cpp-test: build/CMakeCache.txt
	cmake --build build -j
	ctest --test-dir build --output-on-failure

# Leak detection is Linux-only; ASan aborts on macOS when it is enabled.
cpp-test-san:
	cmake -B build-san -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLOBRL_SANITIZE=ON
	cmake --build build-san -j
	ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
	    ctest --test-dir build-san --output-on-failure

bench: build/CMakeCache.txt
	cmake --build build --target bench_latency -j
	./build/bench_latency 200000

bench-py:
	$(PY) scripts/bench_python.py

train:
	$(PY) -m lobrl.train --total-steps 300000 --episode-steps 1000 --out runs/ppo

eval:
	$(PY) -m lobrl.evaluate --ckpt runs/ppo/policy.pt --episodes 30 --episode-steps 1000

# --- multi-agent ---------------------------------------------------------
selfplay:
	$(PY) -m lobrl.selfplay --n-agents 4 --total-steps 300000 --out runs/selfplay_n4

sweep:
	for n in 1 2 4 8; do \
	  $(PY) -m lobrl.selfplay --n-agents $$n --total-steps $$((1200000 / $$n)) \
	        --out runs/selfplay_n$$n; \
	done
	$(PY) scripts/competition_sweep.py

league:
	$(PY) -m lobrl.league --policy ckpt:runs/selfplay_n4/policy.pt@self-play \
	    --policy ckpt:runs/ppo/policy.pt@solo --policy as --policy fixed:3.0 \
	    --episodes 20

dashboard:
	$(PY) scripts/record_episode.py
	$(PY) scripts/record_league.py
	$(PY) scripts/build_dashboard.py

overnight:
	./scripts/overnight.sh 5.85

overnight-stop:
	./scripts/overnight.sh stop

arena-status:
	@cat runs/arena/status.json

perf:
	./scripts/profile_perf.sh

clean:
	rm -rf build runs *.egg-info **/__pycache__
