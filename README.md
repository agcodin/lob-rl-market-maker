# Low-Latency Limit Order Book & Multi-Agent RL Market Maker

A deterministic, zero-heap-allocation limit order book matching engine in C++20,
bound to Python through `pybind11` over contiguous NumPy buffers, plus a PPO
market maker trained against synthetic microstructure flow with an
inventory-penalized Avellaneda–Stoikov reward.

```
cpp/include/lob/      types.hpp  pool.hpp  ring.hpp  book.hpp   # header-only engine
cpp/src/bindings.cpp                                            # pybind11 bridge
cpp/bench/bench_latency.cpp                                     # tick-to-trade harness
python/lobrl/         flow.py env.py ppo.py baselines.py metrics.py train.py evaluate.py
tests/                test_engine.py  test_env.py
scripts/              bench_python.py  profile_perf.sh
```

## Quick start

```bash
make install       # venv + deps + build the extension in place
make test          # 28 engine/env/metric tests
make bench         # C++ tick-to-trade latency percentiles
make bench-py      # bridge throughput + zero-copy verification
make train         # PPO, 300k env steps (~15 min on a laptop CPU)
make eval          # PPO vs fixed-spread and Avellaneda-Stoikov baselines
```

## 1. Matching engine (C++20)

No `std::map`, no `std::list`, no pointer-based trees, no allocation on the hot path.

- **Static object pool** — `StaticObjectPool<Order, 2^18>` is one contiguous arena
  with an index-based free stack. `acquire`/`release` are O(1) pushes and pops on
  an integer stack; exhaustion is reported as a reject, never grown around.
- **Intrusive linked lists** — a price level stores `head`/`tail` *arena indices*.
  A level's FIFO queue is a walk over `int32` indices in one array, so there is no
  pointer indirection and no heap fragmentation.
- **Direct-index price map** — `Level bid_levels_[2^17]` / `ask_levels_[2^17]`,
  indexed straight by the fixed-point tick price. Price lookup is one indexed
  load: no hashing, no bucket collisions, no tree descent. Touch tracking is a
  bounded scan from the last best price when a level empties.
- **Price-time priority** — aggressive orders sweep the opposing book level by
  level, fill the queue head first, decrement level volume, emit one fill event
  per resting order touched, and pop empty levels immediately.
- **Generation-tagged ids** — `order_id = (generation << 20) | slot`. A recycled
  slot bumps its generation, so a stale id can never resolve to the new order.

Measured on an Apple M4 (`make bench`, blocks of 64 operations, ns/op):

| operation | p50 | p99 | p99.9 |
|---|---|---|---|
| limit add (rest) | 3.9 | 7.2 | 32.6 |
| cancel | 5.9 | 9.1 | 11.7 |
| aggressive cross (1 level) | 7.2 | 9.8 | 11.1 |
| L2 snapshot (10 levels) | 7.8 | 10.4 | 11.1 |

All well inside the <100 ns tick-to-trade target. The steady clock granularity on
Apple silicon is ~41.7 ns — coarser than a single book operation — so the harness
times blocks of 64 operations and reports the per-operation cost within a block;
on x86 it reads `rdtsc` directly. `scripts/profile_perf.sh` runs `perf stat` for
IPC and L1-dcache miss rate (Linux only; it prints the `xctrace` equivalent on
macOS).

## 2. Python bridge (`pybind11`)

- `snapshot_into(bid_px, bid_vol, ask_px, ask_vol)` writes top-K prices and
  aggregated volumes **into caller-owned NumPy arrays** that the env allocates
  once at construction. No temporaries, no serialization.
- `events_buffer()` returns a structured NumPy array that *aliases* the C++
  ring buffer, with the book held as the array's base object. Fills, cancels and
  rejects are read in Python without a copy; `events_total()` lets the reader
  detect drops.
- `MarketMakingEnv` is a standard `gymnasium.Env`: `step(action)` cancels and
  re-submits the agent's two quotes, advances the synthetic market by one tick,
  and replays the new ring records into inventory and cash.

## 3. Synthetic market flow

Background traders use the same public order interface as the agent, so the agent
cannot get fills the flow model itself could not.

- **Poisson passive liquidity** — placements arrive at rate `lambda_L`; depth
  behind the touch is geometric, and a small fraction (`p_improve`) betters the
  touch by a tick. That fraction is what closes the spread after a sweep, and it
  is the main knob on the stationary spread.
- **Queue-dependent cancellations** — `lambda_C(d) = alpha / d**beta` per resting
  order, `d` = depth from the best quote, so cancellation accelerates near the
  spread. Only noise orders are ever cancelled.
- **Hawkes aggressive flow** — `lambda(t) = mu + sum_i alpha * exp(-beta (t - t_i))`,
  maintained by exponential-decay recursion, so trades cluster in time.
- **Power-law sizes** — `P(V > x) ~ x^-1.5`, sampled by inverse transform and
  capped, so occasional large sweeps walk several levels.
- **Order-flow feedback** — the sign of aggressive flow leans with book imbalance,
  which is what makes tight quoting adversely selected.

Default calibration: mean spread ≈ 3 ticks, mid-price increment σ ≈ 0.5 ticks/step,
no order rejections over long runs.

## 4. PPO agent

- **State** (`4K + 8` floats, K = 5): top-K bid/ask offsets from mid and
  log-volumes, book imbalance `(Vb - Va)/(Vb + Va)`, inventory `q/Qmax`, spread,
  rolling mid-price volatility, realized and unrealized PnL, and the queue-ahead
  ratio of each of the agent's own live quotes.
- **Action**: `Box(-1, 1, (2,))` mapped affinely to half-spread offsets
  `[delta_bid, delta_ask]` in `[min_offset, max_offset]` ticks around the mid.
  Quotes are clamped so they never cross, and the side that would breach
  `±Qmax` is suppressed.
- **Reward** (Avellaneda–Stoikov utility):

  `R_t = dPnL_t - phi * q_t^2 - eta * (dq_t)^2`,
  `dPnL_t = q_{t-1} (S_t - S_{t-1}) + (Cash_t - Cash_{t-1})`

  with `phi` the inventory risk aversion and `eta` the turnover friction.
- Actor-critic MLPs (2x128, tanh), diagonal Gaussian with state-independent
  log-std, GAE(λ), clipped surrogate objective, KL early stopping, and a Welford
  observation normalizer frozen at evaluation time.

Realized PnL uses average-cost accounting, so closing volume realizes and
opening volume re-averages — the reward itself is mark-to-market, which is the
A-S formulation.

## 5. Verification and benchmarking

- `make test` — matching invariants (price-time priority, partial fills,
  multi-level sweeps, stale-id safety, arena exhaustion, zero-copy event view)
  and environment invariants (determinism under seed, cash/inventory
  consistency, quotes never crossing, inventory cap, flow stationarity, Hawkes
  over-dispersion).
- `make bench` / `make bench-py` — systems latency and bridge throughput.
- `scripts/profile_perf.sh` — `perf stat` counters for IPC and cache behaviour.
- `make eval` — out-of-sample (held-out seed block) annualized Sharpe, maximum
  drawdown, 10-tick adverse-selection markout, mean/max absolute inventory and
  fill counts, for the PPO policy against fixed-spread quoting at 1/3/6 ticks
  and the closed-form Avellaneda–Stoikov quoter.

Markout sign convention: **positive = adversely selected** (the mid moved against
the fill over the next 10 steps).

### Result (30 held-out seeds, 1000 steps/episode, phi = 5e-3, 1.2M training steps)

| policy | PnL | Sharpe | max DD | markout(10) | mean abs inv | fills | win rate |
|---|---|---|---|---|---|---|---|
| fixed 1 tick | 1613 | 20.5 | 1409 | -0.73 | 95.2 | 174 | 0.83 |
| fixed 3 ticks | 1121 | 21.3 | 974 | -1.93 | 53.7 | 49 | 0.80 |
| fixed 6 ticks | 1091 | 21.4 | 622 | -3.62 | 38.7 | 16 | 0.93 |
| Avellaneda–Stoikov | 1708 | 25.8 | 1287 | -0.39 | 92.0 | 384 | 0.80 |
| **PPO** | 801 | **37.7** | **192** | -0.68 | **7.3** | 104 | **1.00** |

The learned policy makes less gross PnL than the baselines and is the better
market maker: it earns from spread capture rather than from carrying inventory,
which is what the risk-adjusted columns show — roughly 1.5x the Sharpe of the
best heuristic at one seventh its drawdown and one thirteenth its average
absolute inventory, positive on every held-out seed. Reproduce with
`make train && make eval`.

## Notes and limitations

- The engine is single-threaded by design; the arena and the price grid are the
  whole state, so a book is trivially snapshot- and replay-able.
- `-mcpu=native` / `-march=native` is opt-in (`LOBRL_NATIVE=1`) because it is
  incompatible with the universal2 build macOS defaults to.
- The market is synthetic. Absolute Sharpe numbers describe this simulator, not
  a real venue; the baselines are there so the comparison is like-for-like.
