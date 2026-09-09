#!/usr/bin/env python
"""Python-side throughput and zero-copy checks for the engine bridge."""

from __future__ import annotations

import time

import numpy as np

from lobrl import OrderBook, Owner, Side
from lobrl.env import EnvConfig, MarketMakingEnv


def bench(fn, n, label):
    fn(1000)  # warm up
    t0 = time.perf_counter()
    fn(n)
    dt = time.perf_counter() - t0
    print(f"{label:<34} {n/dt:12,.0f} ops/s   {dt/n*1e9:9.1f} ns/op")


def main():
    book = OrderBook()
    for d in range(1, 40):
        book.limit(Side.BID, 10_000 - d, 50, Owner.NOISE)
        book.limit(Side.ASK, 10_000 + d, 50, Owner.NOISE)

    bench(lambda n: [book.limit(Side.BID, 9_000, 1, Owner.NOISE) for _ in range(n)],
          200_000, "limit add (through pybind11)")

    bp, bv = np.zeros(10, np.int32), np.zeros(10, np.int64)
    ap, av = np.zeros(10, np.int32), np.zeros(10, np.int64)
    bench(lambda n: [book.snapshot_into(bp, bv, ap, av) for _ in range(n)],
          200_000, "L2 snapshot into NumPy (10 lvls)")

    ev = book.events_buffer()
    print(f"\nevent ring: {ev.shape[0]} slots x {ev.itemsize} B, "
          f"zero-copy view: {ev.base is not None}")
    print(f"dtype: {ev.dtype}")

    env = MarketMakingEnv(EnvConfig(max_steps=10**9), seed=0)
    env.reset(seed=0)
    a = np.zeros(2, np.float32)
    n = 20_000
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(a)
    dt = time.perf_counter() - t0
    print(f"\n{'gym env step (full sim tick)':<34} {n/dt:12,.0f} steps/s "
          f"{dt/n*1e6:9.1f} us/step")


if __name__ == "__main__":
    main()
