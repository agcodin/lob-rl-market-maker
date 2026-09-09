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
                      multi_env.py selfplay.py league.py          # multi-agent
tests/                test_engine.py  test_env.py  test_multi.py
scripts/              bench_python.py  profile_perf.sh  record_episode.py
                      record_league.py  build_dashboard.py  competition_sweep.py
```

## Quick start

```bash
make install       # venv + deps + build the extension in place
make test          # 28 engine/env/metric tests
make bench         # C++ tick-to-trade latency percentiles
make bench-py      # bridge throughput + zero-copy verification
make train         # single-agent PPO, 1.2M steps (~5 min on a laptop CPU)
make eval          # PPO vs fixed-spread and Avellaneda-Stoikov baselines
make sweep         # self-play at 1/2/4/8 makers, then the competition table
make league        # every strategy in one shared order book
make dashboard     # rebuild the HTML dashboard from fresh runs
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

- `make test` — 40 tests: matching invariants (price-time priority, partial
  fills, multi-level sweeps, stale-id safety, arena exhaustion, zero-copy event
  view), environment invariants (determinism under seed, cash/inventory
  consistency, quotes never crossing, inventory cap, flow stationarity, Hawkes
  over-dispersion), and multi-agent invariants (per-owner fill attribution,
  seat fairness, tighter quotes winning more fills, GAE across parallel agent
  streams matching the single-stream reference).
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

## 6. Multi-agent: makers competing in one book

`make sweep` trains a shared policy by self-play at several table sizes, and
`make league` puts different strategies into the same book at once.

- **Engine** — an order's owner is a `uint8`, so one book attributes fills to
  255 distinct participants (0 is the background flow). Fill attribution is
  read straight off the event ring.
- **`MultiAgentMarketMakingEnv`** — simultaneous moves: every maker quotes
  against the *same* pre-step mid, and submission order is reshuffled every
  step. Without that reshuffle whichever agent submits first always wins the
  queue at a shared price, and the ranking measures list position rather than
  skill (`test_submission_order_is_the_thing_that_makes_it_fair` demonstrates the
  bias the flag removes). Observations are byte-for-byte the single-agent
  layout, which is what lets a solo-trained policy and a self-play policy meet
  in the same book.
- **Self-play** — all seats run the same network, so the opposition improves
  exactly as fast as the policy does. The seats still diverge within an episode:
  they hold different inventory and sit in different queue positions.
- **Controlled sweep** — transitions per PPO update are held at 2,048 no matter
  the table size (`rollout = batch // n_agents`). Without that, a bigger table
  silently means a bigger gradient batch and the sweep measures batch size
  rather than competition.

### Head to head, one shared book (32 episodes, seats rotated)

| strategy | PnL | Sharpe | max DD | avg pos | fills | win rate |
|---|---|---|---|---|---|---|
| **New champion (arena)** | 781 | **41.5** | 150 | 11.6 | 160 | 100% |
| Previous champion (self-play) | 511 | 35.5 | 119 | 8.3 | 98 | 100% |
| Solo-trained | 333 | 21.7 | 99 | 5.5 | 33 | 94% |
| Avellaneda–Stoikov | 340 | 5.4 | 1859 | 102.4 | 303 | 62% |
| Fixed 3 ticks | −65 | 2.8 | 628 | 29.0 | 10 | 47% |

`runs/champion.pt` is the current best policy. The earlier four-way table below
is kept for the self-play vs solo-training comparison it makes.

### Earlier: self-play vs solo training (12 episodes, seats rotated)

| strategy | PnL | Sharpe | half-spread | fills | avg pos | win rate |
|---|---|---|---|---|---|---|
| **Self-play PPO** | 664 | **40.8** | 4.95 | 124 | 8.1 | 100% |
| Solo-trained PPO | 383 | 23.4 | 6.47 | 40 | 5.7 | 100% |
| Avellaneda–Stoikov | **794** | 9.4 | 0.66 | 338 | 99.3 | 75% |
| Fixed 3 ticks | 156 | 4.5 | 3.00 | 11 | 33.8 | 58% |

Training under competition produced a better competitor: head to head, the
self-play policy beats the solo-trained one on both raw profit (664 vs 383) and
risk-adjusted return (40.8 vs 23.4). The solo policy quotes wider and takes a
third of the fills — it never learned that a rival will take the queue.
Avellaneda–Stoikov again buys the largest raw PnL with inventory it does not
want (99 shares average, a quarter of the time unprofitable).

### Profit per maker vs table size

| makers | PnL / maker | PnL, all makers | fills / maker | avg pos |
|---|---|---|---|---|
| 1 | 971 | 971 | 116 | 11.4 |
| 2 | 637 | 1275 | 35 | 13.2 |
| 4 | 594 | 2378 | 112 | 8.9 |
| 8 | **144** | 1149 | 25 | 5.7 |

Profit per maker falls at every step — 85% from one maker to eight. **The
quoted-spread column does not trend cleanly and no claim is made about it**:
there is one training seed per table size, and runs land in either a patient
wide-quoting mode or an active tight-quoting one, which moves spread and fill
count together. Separating that from a real effect of competition needs several
training seeds per size, which has not been run.

## 7. Overnight league training

`make overnight` runs an unattended session that improves the policy against a
growing pool of its own past selves. It is plain Python on the local CPU --
no network calls, no external services.

```bash
make overnight          # 5.85 h budget, detached, keeps the Mac awake
make arena-status       # progress while it runs
make overnight-stop     # finishes the current generation, then saves
```

Plain self-play plateaus and can cycle: a policy learns to beat its current
self, forgets what beat its older self, and goes round in circles. The arena
runs prioritized fictitious self-play instead:

- **Mixed opponents** — the learner holds 2 of 4 seats; the others are drawn per
  episode from the live policy (45%), a heuristic quoter (15%), or a snapshot
  from the pool, biased toward recent ones. Old snapshots keep having to be
  beaten, which is what stops the cycling.
- **A promotion gate** — after each generation the candidate plays the incumbent
  over 32 shared-seed episodes with seats rotated, and is only promoted to
  `best.pt` if it actually wins. This is what makes the run monotone rather than
  merely long.
- **A revert valve** — after 8 generations without a promotion the learner is
  reloaded from `best.pt`, so an unattended run cannot spend hours adrift.
- **Bounded footprint** — the snapshot pool is thinned to 60 files (recent kept,
  older subsampled), so a long night does not fill the disk.
- **Resumable** — restarting picks up `best.pt`, the pool, and the generation
  counter.

The inventory penalty in the arena defaults to `2e-2`, four times the
single-agent value. That is a measured setting, not a guess: at `5e-3` the
learner drifts to ~16 average position within minutes and never beats the
incumbent; at `2e-2` it holds ~12 and beats it by +5.6 Sharpe (t = 2.9 over 48
paired episodes). Selection alone cannot fix this -- a correct gate just refuses
everything and the run accomplishes nothing.

### Measured: what actually improves the policy

Four 10-minute arms, each warm-started from the same champion, evaluated in
paired episodes (both policies quoting in the same book on the same seed) on a
held-out seed block, then replicated on a second block:

| arm | change | Sharpe gain | t (block 1) | t (block 2) |
|---|---|---|---|---|
| A | inventory penalty 5e-3 -> 2e-2 | **+5.6** | 3.99 | 2.85 |
| B | drop heuristics from opponent pool | **+6.1** | 4.66 | 2.61 |
| C | learner takes all 4 seats | +1.6 | 0.98 | — |
| A+B | both together | −2.7 | — | −1.41 |

A and B replicate and are statistically indistinguishable from each other
(t = 0.56); combining them does not help. C -- which is closest to plain
self-play with the pool used only for gating -- does nothing, which is the
control showing the arena structure is what matters, not the extra transitions.
Effect sizes shrink from block 1 to block 2 because block 1 was used to pick the
winners; block 2 is the honest estimate.

When the budget expires it runs the test suite, writes
`runs/arena/MORNING_REPORT.md` (progress windows plus a final head-to-head
against every earlier policy and both baselines), and rebuilds the dashboard.

## Notes and limitations

- The engine is single-threaded by design; the arena and the price grid are the
  whole state, so a book is trivially snapshot- and replay-able.
- `-mcpu=native` / `-march=native` is opt-in (`LOBRL_NATIVE=1`) because it is
  incompatible with the universal2 build macOS defaults to.
- The market is synthetic. Absolute Sharpe numbers describe this simulator, not
  a real venue; the baselines are there so the comparison is like-for-like.
- Agent quotes are always passive: the placement clamp keeps them from crossing
  the live book, so makers never take liquidity from one another directly, only
  compete for the same taker flow.
- The self-play runs were still improving at 1.2M transitions (reward trend
  still positive), so the sweep compares four equally-trained but not fully
  converged policies.
- `caffeinate` keeps the Mac awake for the overnight run, but a closed laptop
  lid still sleeps unless an external display is attached.
- `make sweep` trains one seed per table size. The per-maker profit trend is
  monotonic across all four and survives that; the spread numbers do not.
