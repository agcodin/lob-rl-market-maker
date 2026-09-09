PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install bench bench-py test train eval perf clean

venv:
	python3 -m venv .venv && $(PIP) install -U pip wheel setuptools

install: venv
	$(PIP) install -r requirements.txt
	ARCHFLAGS="-arch $$(uname -m)" LOBRL_NATIVE=1 $(PIP) install -e .

test:
	$(PY) -m pytest tests -q

bench: build/bench_latency
	./build/bench_latency 500000

build/bench_latency: cpp/bench/bench_latency.cpp cpp/include/lob/*.hpp
	@mkdir -p build
	c++ -std=c++20 -O3 -DNDEBUG -g -fno-omit-frame-pointer \
	    $$(uname -m | grep -q arm64 && echo -mcpu=native || echo -march=native) \
	    -Icpp/include $< -o $@

bench-py:
	$(PY) scripts/bench_python.py

train:
	$(PY) -m lobrl.train --total-steps 300000 --episode-steps 1000 --out runs/ppo

eval:
	$(PY) -m lobrl.evaluate --ckpt runs/ppo/policy.pt --episodes 30 --episode-steps 1000

perf:
	./scripts/profile_perf.sh

clean:
	rm -rf build runs *.egg-info **/__pycache__
