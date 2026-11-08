# Low-Latency Limit Order Book & Multi-Agent RL Market Maker

[![CI](https://github.com/agcodin/lob-rl-market-maker/actions/workflows/ci.yml/badge.svg)](https://github.com/agcodin/lob-rl-market-maker/actions/workflows/ci.yml)

A deterministic, zero-heap-allocation limit order book matching engine in C++20,
bound to Python through `pybind11` over contiguous NumPy buffers, plus a PPO
market maker trained against synthetic microstructure flow with an
inventory-penalized Avellaneda–Stoikov reward, and a multi-agent league where
several makers compete for the same order flow.

**Headline numbers** — 3.9 to 5.9 ns per order-book operation at the median,
between 3.7 and 16.5 times faster than a `std::map` book doing the same work
depending on the operation, and a learned
market maker that beats the closed-form Avellaneda–Stoikov quoter by 7.7x on
risk-adjusted return while carrying a ninth of the inventory.

```
cpp/include/lob/      types.hpp  pool.hpp  ring.hpp  book.hpp   # header-only engine
cpp/src/bindings.cpp                                            # pybind11 bridge
cpp/bench/bench_latency.cpp                                     # tick-to-trade harness
python/lobrl/         flow.py env.py ppo.py baselines.py metrics.py train.py evaluate.py
                      multi_env.py selfplay.py league.py          # multi-agent
tests/                test_engine.py  test_env.py  test_multi.py
docs/                 EXPERIMENT_LOG.md                          # the full record
scripts/              bench_python.py  profile_perf.sh  record_episode.py
                      record_league.py  build_dashboard.py  competition_sweep.py
```

## Quick start

```bash
make install       # venv + deps + build the extension in place
make test          # 50 Python tests (engine, environment, multi-agent, metrics)
make bench         # C++ latency percentiles against a std::map baseline
ctest --test-dir build  # 36 C++ test cases, ~1.4M assertions
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

### Measured latency

Every operation is also run against `cpp/support/reference_book.hpp`, a
`std::map` of price to a `std::list` of orders with an iterator stored per order
— the obvious implementation, written for clarity. A nanosecond figure on its
own says very little; what it is a nanosecond *against* is the point.

```
machine    : Apple M5 Pro (arm64)
compiler   : clang 21.0.0 (clang-2100.4.1.1)
build flags: -O3 -fno-omit-frame-pointer -mcpu=native
iterations : 200000
```

| operation | engine p50 | engine p99 | engine p99.9 | `std::map` p50 | speedup |
|---|---|---|---|---|---|
| limit add (rest) | **3.91** | 16.94 | 282.17 | 33.86 | 8.7x |
| cancel | **5.20** | 22.12 | 29.30 | 85.94 | 16.5x |
| aggressive cross (1 level) | **5.86** | 7.81 | 10.69 | 21.48 | 3.7x |
| L2 snapshot (10 levels) | **5.86** | 8.45 | 11.06 | 75.52 | 12.9x |

Reproduce with `make bench`, or `./build/bench_latency 200000 --csv` for machine
-readable output. The harness prints the machine, compiler and flags it was built
with, so a pasted result carries its own provenance.

**On the timing method.** Per-operation timing needs a clock that can resolve a
single operation, and on this machine neither can: `cntfrq_el0` advertises 1 GHz,
but the smallest gap the counter actually resolves is ~18 ns, and `steady_clock`
manages ~41 ns. The harness measures that resolution at startup rather than
trusting the advertised frequency, and falls back to timing blocks of 64
operations when it is too coarse, saying so in its output. In that mode the
percentiles describe the per-operation cost *within* a block, which understates
the true tail, because one slow operation is averaged across the block.

Per-operation timing is implemented, but no machine measured so far has earned
it. The x86 CI runner looked like the obvious candidate, and `rdtsc` there
reports 0.409 ns per tick, yet the smallest gap it actually resolves is 9.8 ns,
so that run fell back to blocks as well. The mode is chosen from the measured
resolution at runtime, so a machine with a finer clock will get per-operation
timing without any change here.

Two things the harness refuses to do. It clamps the add phase to the arena's
capacity, because resting more orders than the arena holds turns the tail of the
measurement into rejections, which are far cheaper than a real insert and bias
the median downwards, since a 500,000-iteration run reported 3.25 ns until this
was fixed, against 3.91 ns measured honestly. And it aborts rather than print if
any rejection occurred during the add phase.

**The CI numbers are not these numbers, by design.** CI runs the benchmark as a
smoke test on a shared runner, where the figures are several times slower and
much noisier. The same run that produced the green badge measured the engine at
16.8 to 28.8 ns on an AMD EPYC 7763, against the 3.9 to 5.9 ns in the table
above, which was measured on a quiet laptop with `-mcpu=native`. The speedups
over `std::map` fall too, to between 1.8x and 9.0x. Shared cloud runners are
throttled, virtualized and noisy neighbours, so treat the CI run as proof that
the harness still builds and produces sane output, and the table above as the
measurement.

`scripts/profile_perf.sh` runs `perf stat` for IPC and L1-dcache miss rate
(Linux only; it prints the `xctrace` equivalent on macOS). That script has not
been run, so this README makes no claim about IPC or cache behaviour.

### Testing

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure        # 36 C++ test cases
```

**Unit tests** (`cpp/tests/test_book.cpp`) cover adding, cancelling, partial
fills, price-time priority, crossing and marketable limits, empty-book and price
-grid boundaries, arena exhaustion, snapshot padding, event-ring wraparound, and
id reuse across heavy arena recycling.

**Property and differential tests** (`cpp/tests/test_invariants.cpp`) replay
random order streams and check, after every operation, that:

- volume is conserved exactly — a limit order moves resting volume by its size
  less twice whatever it traded away, a market order by exactly what it filled,
  a cancel by exactly what it was holding;
- the book is never crossed, the touch is always the extreme populated level
  with nothing beyond it, and a level holds orders exactly when it holds volume;
- the engine agrees with the naive reference book on best bid, best ask,
  cumulative volume, live order count, and per-level volume and count;
- queue position matches the reference's FIFO order exactly, which is the
  observable form of time priority.

That is 36 test cases and about 1.4 million assertions. Run them under
AddressSanitizer and UndefinedBehaviorSanitizer with:

```bash
cmake -B build-san -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLOBRL_SANITIZE=ON
cmake --build build-san -j
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-san --output-on-failure
```

Leak detection is enabled in CI, which runs on Linux. It is unsupported on
macOS, where ASan aborts with `detect_leaks is not supported on this platform`,
hence `detect_leaks=0` above.

**What the fuzzing found.** No defect in the engine. Both sanitizers are clean
and the differential test agrees with the reference on every one of the millions
of operations replayed so far. It did catch one bug, in the test itself: the
priority test drew a fresh random order size for each book instead of feeding
both the same one, so the two books diverged and the comparison was meaningless.
Worth recording because an engine bug and a harness bug look identical in a test
report, and only one of them is the engine's problem.

**Known limitation: no self-trade prevention.** The engine matches on price and
time alone and never compares owners, so a participant can trade with itself.
`cpp/tests/test_book.cpp` pins that behaviour in a test tagged
`[known-limitation]` rather than leaving it undefined. Callers that need
prevention enforce it above the engine, which is what the environment does by
clamping its own quotes so they cannot cross. There is also no modify or
replace operation; cancel and re-add is the only path.

**Continuous integration** (`.github/workflows/ci.yml`) builds release and
sanitizer configurations, runs the full C++ suite under both, runs a benchmark
smoke test that asserts every operation produced a finite positive median, and
builds the Python extension to run the engine's binding tests.

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

- `ctest --test-dir build` — 36 C++ test cases and about 1.4M assertions:
  unit tests, plus property and differential tests against a naive reference
  book, all clean under ASan and UBSan. See **Testing** above.
- `make test` — 50 Python tests: matching invariants (price-time priority, partial
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

## 8. The informed market

`informed_frac > 0` turns on Glosten-Milgrom / Kyle informed traders: a latent
fundamental random-walks and a fraction of takers observe it and trade toward it,
dragging the mid with them. This is a **separate benchmark** -- its numbers are
not comparable to the uninformed market above, and it has its own champion
(`runs/champion_informed.pt`; the uninformed champion stays `runs/champion.pt`).

It is a much harder market. Avellaneda-Stoikov goes from +1083 PnL to **-1012**,
and the uninformed champion drops from Sharpe 44.6 to 26.7.

Winning in it needs an estimate of the latent fundamental, and the pipeline is:

1. **Tape features** (`flow_features`) -- signed-volume EWMAs, arrival intensity
   and mean trade size. Informed flow shows up as persistent one-directional
   volume, so this is the only sensor that sees it: book imbalance, which the
   policy already had, predicts the fundamental gap at corr **+0.02**, while the
   tape alone manages **+0.29**.
2. **A supervised estimator** (`scripts/train_gap_predictor.py`, `lobrl/gap.py`)
   of the fundamental gap, from market-visible features only -- never inventory,
   PnL or queue position, so its output is a property of the market and is
   computed once per step for all agents. Held-out corr **+0.48**; a linear model
   caps at +0.29 and reinforcement learning never got past +0.15.
3. **A learnable lean** (`PPOConfig.lean_col`) -- a skip connection from the
   estimate straight to the action mean, so a positive gap pulls the bid closer
   and pushes the ask away.

Why the estimator is supervised rather than learned by RL: the reward gradient
for "read the tape" is indirect and buried in return variance. Measured, PPO left
the tape input weights at 0.0035 (3% of the weight on other state features) after
6M transitions, and the policy's quotes barely responded. Supervised learning on
the same data finds the signal in seconds.

Why the lean is wired in rather than trained from scratch: the behaviour is worth
a measured +6.5 Sharpe (t=5.1) yet PPO did not find it even with the estimate
handed over as an input feature at full weight. Behaviour cloning was tried and
was not precise enough -- a held-out action MSE of 0.001 is ~28% of a lean whose
typical magnitude is 0.087, and it recovered only +1.7 (t=1.1). Wiring it in
exactly and leaving it trainable worked: **+6.17 Sharpe (t=5.33)** over the
baseline, and RL then tuned the lean from the ±0.40 seed to ±0.89/0.76 for a
further **+2.22 (t=2.18)**.

### The estimator must be trained on the market it is used in

`train_gap_predictor.py --driver <policy>` collects its data while a policy is
actually quoting. This is not a detail. An estimator trained on an idle book
scores corr 0.502 there and only **0.406** in the book the champion makes -- the
champion's own quotes change every feature it reads. Retraining on-policy and
swapping *only the estimator*, with the policy weights untouched, measured
**+3.88 Sharpe (t=2.68)** and **+6.21 (t=3.81)** on two independent seed blocks.
One round closes the gap; a second on-policy iteration measured -1.83 (t=-1.01),
so the loop converges immediately and is not worth repeating.

### After a pipeline change, re-sweep the scalars tuned against the old pipeline

Promoting the on-policy estimator changed the gap feature's scale (`gap_sd`
3.225 -> 3.566) and left the lean, which had been tuned against the old one,
too strong. Reinforcement learning did not fix it -- five consecutive training
runs produced zero promotions. A hand sweep did: scaling the learned lean vector
by 0.7 measured **+2.39 Sharpe (t=2.89)** over the champion on 128 paired
episodes. Sweeping a scalar costs a minute; the run that failed to find it cost
five million transitions.

### Change one thing per promotion, or isolate each afterwards

A promotion that swapped the estimator *and* rescaled the lean measured +5.54
Sharpe (t=4.32) against the previous pair, which is real but says nothing about
which half earned it. Tested separately, in the same book with the same
estimator: the lean rescale measured **-1.01 (t=-1.19)** -- not justified -- while
the estimator change measured **+3.55 (t=2.88)**. The whole win was the
estimator; the lean change was reverted, leaving a simpler policy that is at
least as good.

### The agent competes away its own edge

A stronger-leaning market maker pushes the mid toward the latent fundamental
faster, so the gap it profits from shrinks: measured gap standard deviation 4.64
in a weaker champion's market against 3.06 in a stronger one's. Better market
making improves price discovery, and price discovery removes the signal that
made it profitable. This is a real microstructure effect and it caps how far this
particular edge can be pushed.

### Two rules this cost enough to learn

- **Screen on one seed block, confirm on another.** Picking the best of several
  candidates and reporting its screening statistic is the classic trap: the
  screen chose lean scale 0.7 at +4.23, and the confirmation on a fresh block
  measured +2.39. Only the second number is real.
- **Evaluate in pairs.** Both policies quote into the same book on the same seed
  with seats alternating. Unpaired, the noise is +-4 Sharpe and swamps every real
  effect here. And size the sample to the effect: a stable +2.2 read as
  non-significant three times at n=64 before n=160 resolved it.
- **A ceiling probe is only valid if the thing it simulates is reachable by the
  mechanism the agent actually has.** The oracle probe (change only the quotes)
  was sound and predicted the +6 win. A later probe measured the PnL of holding a
  position proportional to the estimate, found an annualised Sharpe of 48, and was
  wrong -- it assumed the position was free to hold, when the agent has to acquire
  it through fills and pay spread and adverse selection for it. Training against
  that target measured **-4.30 Sharpe (t=-4.72)**. The lean was already the
  efficient way to express the same view, because there the position accumulates
  as a byproduct of spread capture at no extra cost.

## Where to read the details

`docs/EXPERIMENT_LOG.md` is the verbatim record of the session that produced the
informed-market champion: 15 checkpoints, every measurement, and the reasoning at
each decision including the wrong turns. It is kept unedited because the failures
carry most of the transferable content -- an overnight run that spent 103M
transitions making the agent worse, a promotion gate that promoted on noise for a
whole session, a ceiling probe that was invalid in a way that took a -4.30 Sharpe
result to expose.

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
- The informed market is a separate benchmark with its own champion; do not
  compare its numbers to the uninformed ones.
- `make sweep` trains one seed per table size. The per-maker profit trend is
  monotonic across all four and survives that; the spread numbers do not.
