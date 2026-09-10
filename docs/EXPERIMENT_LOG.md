# Autonomous experiment log

User is out; I run experiments, check every 30 min, pivot when an arm stalls.
Append one block per check-in. Newest at the bottom.

## Standing facts

- Champion: `runs/champion.pt` (2-d action, hidden 128). Beat previous by +5.6
  Sharpe (t=2.85). Two subsequent 20-min runs FAILED to improve it -> it is near
  a local optimum for the plain 2-d arena. Do not just run that config longer.
- Evaluation MUST be paired (both policies in the same book, same seed, seats
  alternating) or the +-4 Sharpe noise swamps everything.
- Significance bar: |t| >= 2 on >= 48 paired episodes. Taking the max over many
  snapshots is a multiple-comparisons trap -- max |t| from 14 pure-noise draws
  has median ~2.0, so a lone t=2.1 from a scan means nothing.
- Never overwrite `runs/champion.pt` unless a fresh-seed paired test clears the
  bar. Keep a dated copy before replacing it.
- Seed blocks used so far (do not reuse for a new decision):
  70k, 80k (arm selection), 90k, 95k, 120k, 130k, 140k.

## Ideas queue

1. [RUNNING] Variable quote size -- action 2-d -> 4-d [db, da, size_b, size_a].
   Real capability gap: the champion cannot size its quotes at all.
   - `exp_size`: expanded champion, hidden 128
   - `exp_big` : expanded champion, hidden 256 (capacity too)
   Both expansions verified EXACTLY function-preserving at init (max action
   diff 0.00e+00), so they start at champion parity.
2. [QUEUED] Informed traders in the flow: a taker who sees the future mid and
   trades on it. Makes adverse selection a real adversary. Note this changes the
   environment, so it breaks comparability with every number above -- would need
   its own champion and its own baselines.
3. [QUEUED] Asymmetric size skew as the only new action (cheaper than 1 if 1
   fails for optimisation rather than capability reasons).
4. [QUEUED] Longer episodes (1000 -> 4000 steps) so inventory management over a
   longer horizon actually matters.

## Check-ins

### t=0 launch
- Implemented variable_size in `MultiEnvConfig` (action 4-d, size mapped onto
  [2,40], action 0 -> 20 shares so 2-d policies are a strict subset).
- `scripts/expand_policy.py` widens a trained head function-preservingly.
- Launched exp_size (seed 11) and exp_big (seed 12), 8h budget each, 3 threads
  each. Will stop them when an arm is decided.
- 45 tests passing.

### Check-in 1 (t+5 min) — too early for a verdict, both arms healthy
- exp_size: 2 generations, 1 promotion. exp_big: 2 generations, 0 promotions.
- Both processes alive at ~100% CPU, 8h budget remaining.
- No paired evaluation run: 0.3M transitions is far too little to move a policy
  that took ~3.7M to build. First real measurement at check-in 2.
- Cron job 00b07b29 fires every 30 min. NOTE: the cron is session-only, but the
  two training runs are detached (nohup + caffeinate, 8h budgets), so training
  continues even if the session ends.

### Check-in 2 (t+35 min) — idea 1 KILLED, pivoted to a harder market

**Measured (paired, fresh seed block 150k, 48 episodes, variable-size env):**
| arm | Sharpe | champ | diff | stderr | t |
|---|---|---|---|---|---|
| exp_size | 24.8 | 23.8 | +1.00 | 1.62 | 0.62 |
| exp_big  | 22.3 | 25.5 | -3.16 | 1.59 | -1.98 |

Neither clears |t|>=2. Diagnostic that mattered more than the verdict: mean
quote size was still 18.5-20.6 shares after 1.5M transitions, i.e. the agents
barely touched the new dimension.

**Decisive test instead of waiting another 30 min:** hand-crafted inventory-based
size skew on top of the champion (an upper bound on what could be learned).
k=0.3 -> +2.79 (t=1.44), k=0.6 -> -0.51 (t=-0.33), k=1.2 -> +3.43 (t=1.77).
Non-monotonic, none significant => **the size dimension has no value in this
market**. Both arms killed. Idea 1 and idea 3 are both refuted by this.

**Idea 5 (new): tape/order-flow features.** Added 4 EWMA features (signed volume
fast/slow, arrival intensity, mean trade size). Then tested predictiveness
BEFORE training (the check I should have run on size):
  uninformed market, corr with forward mid move at +25 steps:
    signed vol fast +0.004, slow -0.117, intensity +0.060
    book imbalance (ALREADY observed) +0.031, and +0.137 at +1 step
  => the tape is a noisy echo of imbalance, which causes trade signs in this
     flow model. Refuted too: no observation gap either.

**Conclusion: the old market is saturated.** 4 straight failures (20-min rerun,
exp_size, exp_big, hand-crafted skew) plus a refuted observation gap.

**Pivot -> idea 2, informed traders (Glosten-Milgrom / Kyle).** A latent
fundamental random-walks; `informed_frac` of takers observe it and trade toward
it. Verified:
- mid tracks the fundamental (corr 0.88 at frac=0.35, 0.99 at 0.60)
- the tape BECOMES predictive: signed vol slow +0.198 at +25 steps vs
  imbalance's +0.099, and the tape's edge GROWS with horizon while imbalance's
  decays -- the signature of informed flow
- the market is genuinely harder for everyone:
    champion Sharpe 44.6 -> 26.7, PnL 914 -> 487
    A-S      Sharpe 15.4 -> -9.1, PnL 1083 -> **-1012** (loses money)
    fixed 3t Sharpe 17.1 -> 10.8

**NEW BENCHMARK.** informed_frac=0.35, fundamental_vol=0.35, flow_features=True.
Numbers here are NOT comparable to anything above. `runs/champion.pt` remains
the champion of the OLD market and is untouched.

Launched from the champion expanded to 32-dim obs (verified function-preserving:
max action diff 0.00e+00 even with random tape values, so it starts wired to the
tape but ignoring it):
- `inf_base`: hidden 128, seed 21
- `inf_big` : hidden 256, seed 22
Baseline to beat in the new market: expanded champion, Sharpe ~26.7.
Seed blocks now burned: 70k 80k 90k 95k 120k 130k 140k 150k 160k 170k.
47 tests passing.

### Check-in 3 (t+65 min) — found the real headroom; retargeted the aux task

**Measured (paired, fresh block 180k, 48 eps, informed market):**
| arm | Sharpe | base | diff | stderr | t |
|---|---|---|---|---|---|
| inf_base | 14.7 | 14.2 | +0.57 | 1.30 | 0.44 |
| inf_big  | 14.1 | 14.0 | +0.07 | 1.36 | 0.05 |

No improvement after 6M transitions. **Diagnosis:** the tape input weights grew
from exactly 0.0000 to only 0.0035 (3% of the weight on the other state
features); action sensitivity to the tape = 0.0036, i.e. ~zero. PPO cannot find
the signal because the benefit is indirect and swamped by return variance.

**Ceiling probes (measure before building — this is now the house rule):**
- Best possible prediction of FORWARD RETURN from the full 32-d observation:
  ridge corr +0.155, R2 0.024. Edge ~0.22 ticks against a ~5-tick spread =>
  an aux task on forward returns has nowhere to go. Confirmed empirically: aux
  head plateaued at corr +0.054.
- **ORACLE PROBE** — give two seats the true fundamental and let them lean:
  k=0.5 -> +24.4 Sharpe (t=10.2); k=1.5 -> +32.3 (t=9.3). Oracle Sharpe 47 vs
  blind 15. **So the informed market has enormous headroom** — the earlier
  failures were about the target, not the market.
- Predicting the FUNDAMENTAL GAP (a persistent level, not a difference):
    tape only      corr +0.292  R2 0.085   <- the only useful sensor
    book only      corr +0.129
    imbalance only corr +0.022  <- worthless here, and it is what the agent had
  3.5x more predictable than forward returns.

**Change:** aux head retargeted from forward return to the fundamental gap,
aux_coef 0.25 -> 2.0. The gap is a training-time-only label (env info), never an
observation — the policy still trades on what it can see. Verified: aux_loss
0.54 -> 0.17 and the head predicts the gap at corr **+0.323**, above the +0.29
linear ridge ceiling.

**Launched a clean A/B** (both from the expanded champion, informed market):
- `inf_aux`  : aux_target=gap, aux_coef=2.0, seed 31
- `inf_noaux`: aux_coef=0.0 (control), seed 32
This isolates whether the auxiliary task is what helps.
Baseline: expanded champion, ~14.2 Sharpe in a 4-strong-seat book.
Seed blocks burned: ...180k 190k. 47 tests passing.

**Refuted so far (all by direct measurement, not by burning training time):**
1. variable quote size — hand-crafted upper bound not significant
2. tape features in the OLD market — less predictive than existing imbalance
3. aux task on forward returns — ceiling R2 0.024

### Check-in 4 (t+95 min) — aux protects but does not improve; found a +8 Sharpe lever

**Measured (paired, fresh block 200k, 48 eps, informed market):**
| comparison | Sharpe | ref | diff | stderr | t |
|---|---|---|---|---|---|
| inf_aux vs baseline | 11.6 | 12.3 | -0.75 | 1.53 | -0.49 |
| inf_noaux vs baseline | 7.0 | 11.4 | **-4.42** | 1.39 | **-3.18** |
| inf_aux vs inf_noaux | 11.6 | 10.0 | +1.68 | 1.58 | 1.06 |

The aux task WORKS mechanically -- tape input weight 0.0000 -> 0.0299 (27% of the
other state weights) vs 0.0090 for the control, a 3.3x difference. And training
in the informed market DEGRADES a policy (control -4.42, t=-3.18) while the aux
arm holds flat. So the aux task protects, but does not improve.

**Why:** the trunk carries the signal but the policy barely acts on it --
corr(action skew, true gap) only +0.115. PPO will not discover "lean on this"
from an indirect reward.

**Fix: stop asking RL to learn it.** Trained a SUPERVISED estimator of the
fundamental gap from market-visible features only (no inventory/PnL/queue, so
one estimate per step shared by all agents):
  held-out corr **+0.476** (linear ridge ceiling +0.29; RL managed ~0.15)
Appended its output to the observation as feature 32 (obs 32 -> 33). In-env the
feature tracks the true gap at corr +0.418.

**Ceiling probe on the REALISTIC signal** (lean on the estimate, not the truth):
| lean k | leaning | blind | diff | t |
|---|---|---|---|---|
| 0.0 | 13.1 | 13.4 | -0.32 | -0.16 |
| **0.4** | **23.6** | 15.5 | **+8.17** | **4.50** |
| 0.8 | 19.0 | 17.0 | +2.04 | 1.02 |
| 1.5 | 21.4 | 13.3 | +8.14 | 2.81 |

**+8 Sharpe is available and learnable.** Strongest positive result of the run.

**Launched** (both from the champion expanded to 33-d, verified function-preserving):
- `gap_feat`: predictor feature + aux(gap) coef 2.0, seed 41
- `gap_only`: predictor feature, aux off (control) -- does the feature alone suffice?
Both promoted on generation 1 (+2.56, +2.58), which the previous arms never did.
Seed blocks burned: ...200k 210k. 47 tests passing.

### Check-in 5 (t+125 min) — **FIRST REAL WIN**: +6.17 Sharpe, t=5.33, PROMOTED

**Measured (paired, fresh block 220k, 48 eps):**
| arm | diff vs 33-d baseline | t |
|---|---|---|
| gap_feat | -0.23 | -0.17 |
| gap_only | -2.74 | -2.09 |

Neither RL arm improved, even though gap_feat read the feature at full weight
(0.0942, on par with the other state features). Note: corr(action skew,
predicted gap) is a CONFOUNDED diagnostic -- the pristine baseline scores +0.330
with a feature weight of exactly 0.0000, because skew and the gap share book-state
causes. Use weight magnitude, not that correlation.

**Two things established first:**
- the lever replicates on a fresh block: +6.49, t=5.10
- when ALL FOUR seats lean, mean Sharpe rises 11.5 -> 21.8, so it is a genuine
  collective defence against informed flow, not zero-sum exploitation

**Behaviour cloning (`scripts/clone_policy.py`): tried and insufficient.**
Held-out action MSE 0.00112 sounds tiny but is ~28% of a lean whose typical
magnitude is 0.087. Result: only +1.73 (t=1.11). Recorded as a negative.

**What worked: wire the behaviour into the architecture.** `PPOConfig.lean_col`
adds a learnable skip from the predicted-gap column straight to the action mean,
initialised to the measured-best k=0.4.
*Bug found and fixed on the way:* the base head saturates past the action bound,
so `shift-then-clip` swallowed the bid leg entirely (it moved 0.0 instead of
0.2). Clamping BEFORE adding the lean makes it exact -- verified max action diff
vs the hand-crafted policy = 0.00e+00.

**PROMOTION (64 paired episodes, fresh block 250k):**
    lean Sharpe 16.4, PnL 358, |inv| 12.2, maxDD 269
    base Sharpe 10.2
    diff **+6.17**, stderr 1.16, **t=5.33**, 95% CI [+3.90, +8.44]
-> `runs/champion_informed.pt` is the new informed-market champion.
   `runs/champion.pt` (old market) untouched, backed up dated.

**Launched RL refinement from the new champion** (the lean vector is trainable,
so RL can tune k and make it state-dependent):
- `lean_rl` : lean + aux(gap), seed 51
- `lean_rl2`: lean, aux off, seed 52
Seed blocks burned: ...220k 230k 240k 250k. 47 tests passing.

### Check-in 6 (t+155 min) — RL is refining the lean; predictor upgrade measured and declined

**Measured (paired, fresh block 260k, 64 eps, vs the promoted champion):**
| arm | Sharpe | champ | diff | stderr | t |
|---|---|---|---|---|---|
| lean_rl (lean + aux)  | 26.5 | 24.4 | +2.02 | 1.39 | 1.45 |
| lean_rl2 (lean, no aux) | 22.7 | 21.9 | +0.80 | 1.40 | 0.58 |

Neither clears |t|>=2 yet, but **RL is actively using the mechanism**: the lean
vector moved from its ±0.40 init to ±0.61/0.58 (lean_rl) and ±0.68/0.80
(lean_rl2). Both arms independently pushed it UP, i.e. leaning harder pays.
Absolute Sharpe is also much higher than earlier evals (26.5 / 24.4 vs 16.4 /
10.2) because now BOTH sides of the pairing lean -- consistent with the earlier
"everyone leans -> 11.5 to 21.8" collective result. Kept running; +2.02 trending
positive with a still-moving lean is worth another window, not a pivot.

**Predictor upgrade: measured, then declined.**
Offline, observation history helps the gap estimate: snapshot corr 0.525 ->
0.564 with tape at 5 lags (R2 0.275 -> 0.318). Implemented it as extra
signed-volume EWMA timescales (recursive, no ring buffer) and retrained:
corr 0.476 -> **0.493** only. Gain appears to scale with R2, so that is ~+0.5
Sharpe -- and it widens the observation, which would invalidate both running
arms AND the promoted champion. **Declined**: default restored to two
timescales, capability kept behind `MultiEnvConfig.flow_alphas`, measurement
recorded here for a future benchmark generation.
(Caught a real hazard doing this: the running arms hold the OLD module in
memory, so evaluating their checkpoints against edited feature code would have
silently mismatched the observation width. Restored before any evaluation.)

Also fixed: `scripts/train_gap_predictor.py` had a duplicate local GapNet left
over from the refactor into `lobrl/gap.py`. Now imports from the package.
Seed blocks burned: ...260k. 47 tests passing.

### Check-in 7 (t+185 min) — **SECOND PROMOTION** (+2.22, t=2.18); reward found mis-specified

**Promotion.** lean_rl showed +2.02 (t=1.45, block 260k) then +2.26 (t=1.34,
block 270k) -- consistently positive, never clearing at n=64. Instead of
guessing, ran a larger decisive test:
  **160 paired episodes, fresh block 280k: +2.22, stderr 1.02, t=2.18,
  95% CI [+0.23, +4.21]** -> CLEARS.
Three independent blocks give +2.02 / +2.26 / +2.22 -- a very stable effect that
n=64 simply could not resolve. `runs/champion_informed.pt` updated (previous
backed up dated). RL had tuned the lean from the ±0.40 seed to **[-0.89, +0.76]**.
Killed lean_rl2 (+1.76, t=1.15) to free cores; lean_rl left running.

**Gap-predictor ceiling: near exhausted.** With 150k samples and history:
    snapshot                     corr 0.495  R2 0.245
    snapshot + tape at 6 lags    corr 0.534  R2 0.286
    snapshot + FULL state, 6 lags corr 0.543  R2 0.295
~0.54 looks close to the information limit (65% of flow is uninformed noise).
Worth ~+1 Sharpe. Not pursued further.

**The bigger find: the REWARD is mis-specified for this market.**
Avellaneda-Stoikov penalises q^2 -- all inventory treated as risk. That is only
correct for a martingale mid. Measured directly:
    a position q = k * predicted_gap earns **annualised Sharpe 48.2** on its own
    (pure directional PnL, no spread capture), at every k tested
    but phi*q^2 at |q|=20 costs 8.0/step while the drift edge earns 0.47/step
    -> **the penalty outweighs the edge by ~17x; the agent is punished for
       being right.**

**Change:** `inventory_target_gain` makes the penalty phi*(q - q*)^2 with
q* tracking the predicted gap. Unwanted inventory is still penalised; a
justified position is not. This changes the TRAINING reward only -- evaluation
is Sharpe of equity, which is reward-independent, so all comparisons stay valid.
Test added asserting equity is identical across the two reward settings while
q* differs.

Launched `inv_target` from the new champion (gain=40, chosen from the probe's
k=40 row). Promoted on generation 1. `lean_rl` continues as the control.
Seed blocks burned: ...270k 280k. 48 tests passing.

### Check-in 8 (t+215 min) — inventory-target reward REFUTED; my probe was flawed

**Measured (128 paired episodes, fresh block 290k, standard eval env):**
    inv_target Sharpe 23.9, PnL 532, |inv| 12.3, maxDD 221
    champion   Sharpe 28.2
    diff **-4.30**, stderr 0.91, **t=-4.72** -> significantly WORSE.

**Why the probe misled me.** It measured `sum q_t * (mid_{t+1}-mid_t)` for
q = k*predicted_gap and found annualised Sharpe 48. That assumes the position is
FREE TO HOLD. It is not: the agent has to acquire it through fills, paying
spread and adverse selection, and the realised position lags the signal. The
earlier oracle probe was sound because it only changed QUOTES, which cost
nothing extra; this one silently assumed away the entire cost of trading.
**Rule added: a ceiling probe is only valid if the thing it simulates is
actually reachable by the mechanism the agent has.**

**The useful conclusion:** the lean is already the efficient way to express the
view -- the position accumulates as a byproduct of spread capture, at no extra
acquisition cost. Explicit position-taking adds cost without adding edge. The
alpha channel is well exploited; that is why lean worked and this did not.
Reverted (`inventory_target_gain` defaults to 0.0, capability kept + tested).

**lean_rl also stalled**: 18.5M transitions, 12 reverts, no promotion beyond the
one already taken. Killed.

**New plan: seed variance.** The config is good and saturated for a single seed,
so run the SAME best config under three seeds (71/72/73) and validate the best
on a held-out block with n>=128. Choosing among seeds is a legitimate way to
improve provided the winner is confirmed out-of-sample -- and note the
multiple-comparisons hazard: picking the max of three then reporting its
in-sample t would be exactly the trap from the overnight run, so the winner gets
a fresh block.
Seed blocks burned: ...290k. 48 tests passing.

**Refuted this session (6):** variable quote size; tape features in the old
market; aux on forward returns; behaviour cloning; multi-timescale predictor
(worth only +0.5, declined); inventory-target reward.
**Won (2):** gap-lean skip connection (+6.17, t=5.33); RL-tuned lean (+2.22, t=2.18).

### Check-in 9 (t+245 min) — seed sweep NEGATIVE, and it exposed a flaw in the gate

**Stage 1 screen (block 300k, n=64), all three seeds vs the champion:**
    seed71 -5.80 (t=-4.23) | seed72 -7.38 (t=-4.89) | seed73 -2.15 (t=-1.73)
All three WORSE. No stage-2 confirmation run, no promotion.

**Why that is diagnostic, not just disappointing:** every arm STARTS from the
champion and its internal promotion gate is supposed to stop it regressing. It
did not. Looking at seed73's history:
    promotions were granted at gains of **+0.72, +0.20, +2.33**
    while the sd of the gain series -- the gate's own noise -- is **3.33 Sharpe**
So the gate has been promoting on `gain > 0` with no allowance for noise. A
candidate that is genuinely ~2 Sharpe worse clears a single 32-episode check
whenever noise favours it, and over 40 generations that leaks reliably. That is
precisely the 2-7 Sharpe regression seen across all three seeds, and it has been
true of every arm this session.

**Fixed.** `head_to_head` now returns the standard error of the per-episode
paired difference, and the gate requires
    gain > max(promote_margin, noise_k * paired_se)   (noise_k default 1.0)
The log line now prints `gain/threshold`. Two tests added: one asserting
head_to_head reports a finite paired_se, one asserting a +0.2 gain cannot clear
a 3.3 noise sd while a +5.0 gain still can.
(The first version of that test was degenerate -- random policies over 120 steps
never fill, so every Sharpe was 0 and the sd was legitimately 0. Rewritten.)

**Relaunched** gate81/82/83 from the champion with noise_k=1.0 and
eval_episodes=48. The question for next check-in is narrow and worth answering:
does the fixed ratchet stop the regression?
Seed blocks burned: ...300k. 50 tests passing.

### Check-in 10 (t+275 min) — gate fix CONFIRMED; champion converged; dashboard updated

**The narrow question from last check-in is answered: the ratchet now holds.**
- gate81 and gate83: **0 promotions in 32 generations, and their best.pt is
  byte-for-byte identical to the champion.** Before the fix, the same setup
  ended 2-7 Sharpe BELOW it.
- gate82: 1 promotion, and it cleared its threshold honestly (+1.86 vs 1.31).
  Measured on a fresh block (310k, 96 paired eps): **+0.79, t=0.87** -- safe but
  not significant. No promotion.

Across all three arms the fixed gate allowed 1 promotion where the old gate
would have allowed ~10. Training is now safe by construction: it can no longer
quietly erode the champion.

**Conclusion: the champion is converged for this configuration.** Four
independent attempts to improve it (inv_target, seed71/72/73, gate81/82/83) all
failed to clear, and three of them regressed under the old gate. Arms left
running -- they cannot do harm now -- but the expectation is no further gain
without a structural change.

**Consolidated into the dashboard** (`scripts/record_informed.py` + a new
section). The headline measurement is a clean specialisation result, 32 episodes
per market:
| strategy | informed mkt | uninformed mkt |
|---|---|---|
| Informed champion | **30.6** | 26.7 |
| Uninformed champion | 26.5 | **43.2** |
| Avellaneda-Stoikov | **-8.0** (16% win, 108 shares) | 10.5 |
| Fixed 3 ticks | 7.9 | 5.0 |
Each champion wins its own market and loses the other. A-S loses money outright
once adverse selection is real.
(Caught a stray `function标(){}` typo in the template before publishing -- the
node --check step earns its keep.)

Seed blocks burned: ...310k, 400k (standings). 50 tests passing.

### Check-in 11 (t+305 min) — **THIRD WIN**: on-policy estimator, +3.88 and +6.21

**gate81/82/83 stalled** (0/1/0 promotions, all gains now negative against
threshold). Killed. The fixed gate did its job: no regression, best.pt still the
champion.

**Found: the estimator was being used off-distribution.** It was trained on a
market where nobody quotes, but it is USED in the market the champion makes:
    zero-action market (trained on): corr **0.502**
    champion's market (used in):     corr **0.406**   <- 19% worse
A market with no market maker is not the market the estimate is for.

**Fix:** `train_gap_predictor.py --driver <policy>` collects data while a policy
quotes. Retrained on champion-generated data (v3): corr **0.457** on the
distribution it is actually used on, vs 0.406 for v1.

**Measured, same champion policy, only the estimator swapped** (paired by seed,
common random numbers):
    block 320k, 96 seeds: v3 32.7 vs v1 28.8, diff **+3.88**, t=**2.68**
    block 330k, 80 seeds: v3 32.2 vs v1 26.0, diff **+6.21**, t=**3.81**
Two independent blocks, both significant. **PROMOTED**: `runs/gap_predictor.pt`
is now v3, v1 backed up dated. The policy weights are unchanged -- this win is
entirely in the estimator's training distribution.

**A second on-policy iteration (v4) does NOT help**: -1.83, t=-1.01 vs v3. One
round closes the gap; the loop has converged. Recorded so nobody iterates it
again expecting more.

Arms restarted on the promoted estimator (onpol91/92) so nothing holds a stale
copy in memory -- the same hazard flagged at check-in 6.
Seed blocks burned: ...320k 330k 340k. 50 tests passing.

**Session totals: 3 wins** (+6.17 lean, +2.22 RL-tuned lean, +3.88/+6.21
on-policy estimator), **8 refutations**, **1 machinery bug fixed** (promotion
gate promoting on noise).

### Check-in 12 (t+335 min) — **FOURTH WIN**: recalibrating the lean, +2.39 (t=2.89)

**onpol91/92 stalled**: 0 promotions each, best.pt byte-identical to the
champion. That is the **fifth consecutive failure of RL to improve this policy**
(inv_target, seed71-73, gate81-83, onpol91-92). The gate protects it, but policy
gradient has nothing left here. Killed.

**The insight that paid: promoting the estimator left the lean mis-calibrated.**
The lean was tuned against v1 (gap_sd 3.225); the promoted v3 has gap_sd 3.566,
so the feature scale shifted underneath it and RL never retuned it.

Swept a scale multiplier on the champion's lean vector (block 350k, n=48):
    0.7x +2.42 | 1.0x +0.08 | 1.3x -3.28 | 1.6x -4.19
(the 1.0x row is a harness sanity check -- a policy against itself, +0.08 as it
should be). Better estimator wants a SMALLER lean, not a larger one.

Two-stage, done properly:
    screen (block 360k, n=64): 0.40 +0.26 | 0.55 +3.34 | **0.70 +4.23** | 0.85 +1.41
    confirm (block 370k, n=128): **+2.39, stderr 0.83, t=2.89, CI [+0.77,+4.01]**
The screen said +4.23 and the confirmation +2.39 -- textbook winner's-curse
shrinkage, which is why the screen's number is never the one reported.
**PROMOTED**: lean [-0.89,+0.76] -> [-0.62,+0.53]. Previous backed up dated.

**Standing lesson: after any pipeline change, re-sweep the scalars that were
tuned against the old pipeline.** Three of this session's four wins came from
engineering the pipeline (wire the behaviour in, fix the training distribution,
recalibrate) and only one from RL itself.

Relaunched recal101/102 from the recalibrated champion -- the operating point
moved, so it is worth one window to see if RL can now find something.
Seed blocks burned: ...350k 360k 370k. 50 tests passing.

### Check-in 13 (t+365 min) — **FIFTH WIN**: estimator history, +5.54 (t=4.32)

**recal101/102 stalled** (0 promotions, best.pt byte-identical). Sixth
consecutive RL failure. Killed.

**Lean functional form: REFUTED.** Swept deadbands and power transforms on the
lean (block 380k, n=48):
    linear (control) +1.48 | deadband .10 +1.17 | .20 +1.83 | .30 -4.35
    power 0.5 (amplify small estimates) -5.93 | power 2.0 -0.35
The control row is a policy against ITSELF and reads +1.48, so noise at n=48 is
~±1.4 and none of the small positives mean anything. Linear is fine. (Including
that control is what stopped me reading +1.83 as a result.)

**What worked: give the ESTIMATOR history without widening the observation.**
The lag stack lives inside the estimator; the policy still sees one scalar, so
the observation width and every existing champion stay valid.
    on-policy snapshot estimator: corr 0.457
    on-policy + lags (2,4,8,16,32): corr **0.598**   (R2 0.209 -> 0.357, +71%)
Better than the 0.543 offline ceiling I had estimated, because that was measured
on zero-action data.

A better estimator wanted a slightly STRONGER lean this time (the opposite of
last check-in, where better calibration wanted a weaker one -- so re-sweep, do
not extrapolate):
    screen (390k, n=64): 1.0x +1.51 | **1.3x +2.60** | 1.6x +2.11
    confirm (410k, n=128): **+5.54, stderr 1.28, t=4.32, CI [+3.03,+8.06]**
**PROMOTED**: estimator -> history version, lean [-0.62,+0.53] -> [-0.81,+0.69].
New champion: Sharpe **36.4**, PnL 825, |inv| 12.9 (was ~30.9).

Relaunched hist111/112 from the new champion.
Seed blocks burned: ...380k 390k 410k. 50 tests passing.

**Session: 5 wins, 9 refutations, 1 machinery bug.** Every win came from the
pipeline (wire the behaviour in; train the estimator on-policy; recalibrate;
give the estimator memory) except one, where RL tuned a scalar that was already
wired in.

### Check-in 14 (t+395 min) — CORRECTED last check-in's promotion; estimator win confirmed

**hist111/112 stalled** (0 promotions, byte-identical). Seventh consecutive RL
failure. Killed.

**Estimator is exhausted.** Collected fresh on-policy data from the new champion
and swept variants offline -- none beats the current design:
    lags 2-32 h96 (current) 0.549 | h192 0.542 | lags 2-128 0.537 | fib 1-89 0.546
Retraining on the new champion's market gains ~0.011 over the promoted
estimator (0.538 -> 0.549). Not worth a promotion.

**I made an attribution error last check-in and caught it.** That promotion
changed TWO things at once -- the estimator AND the lean (1.3x) -- and the
confirmation compared the new pair against the old pair. It never isolated
either. Tested properly, both in the same book with the same estimator:
    lean 1.3x vs 1.0x, 128 paired eps, block 420k: **-1.01, t=-1.19** -> NOT
    justified, possibly slightly harmful.
    estimator history vs snapshot, same policy, 112 eps, block 430k:
    **+3.55, stderr 1.23, t=2.88, CI [+1.13,+5.96]** -> the whole win.
**Reverted the lean to [-0.62,+0.53]; kept the history estimator.** The champion
is now the simpler policy and is at least as good.
**Rule: change one thing per promotion, or isolate each afterwards.** The +5.54
headline was real but I had credited it to the wrong half.

**A genuine finding: the agent competes away its own edge.** A stronger-leaning
champion pushes the mid toward the fundamental faster, so the exploitable gap
shrinks -- gap sd 4.64 in the weaker champion's market vs 3.06 in the stronger
one. Better market making improves price discovery, which removes the very
signal it profits from. That is a real microstructure effect, not an artifact.
(Careful: the per-champion lineage table I first ran was SINGLE-seed and its
Sharpe column was pure noise -- it is only usable for the gap statistics, which
are averages over 1000 steps within an episode.)

Relaunched att121/122 from the corrected champion.
Seed blocks burned: ...420k 430k. 50 tests passing.
**Session: 5 wins, 10 refutations, 1 machinery bug, 1 self-correction.**

### Check-in 15 (t+425 min) — benchmark saturated; consolidated and stopping the loop

**att121/122 stalled** (0 promotions, byte-identical). **Eighth consecutive RL
failure.** Killed.

**Final standings, informed market, 48 episodes, seats rotated:**
| strategy | Sharpe | PnL | avg pos | maxDD | win |
|---|---|---|---|---|---|
| **Informed champion** | **34.8** | 644 | 11.9 | 140 | 100% |
| Uninformed champion | 25.0 | 539 | 13.0 | 205 | 94% |
| Avellaneda-Stoikov | **-10.5** | -1264 | 114.6 | 2521 | 23% |
| Fixed 3 ticks | 7.1 | 166 | 33.6 | 595 | 62% |
In the uninformed market the two champions reverse (41.3 vs 30.8): each is a
specialist.

**Why I am stopping the 30-minute loop rather than continuing:**
- 8 consecutive RL runs, ~40M transitions, zero promotions
- estimator: architecture sweep found nothing; on-policy retrain gains 0.011
- lean: scale swept twice, functional form (deadband/power) refuted
- and the edge is self-limiting -- a better maker improves price discovery,
  shrinking the gap it profits from (gap sd 4.64 -> 3.06)
Every avenue that produced a win is measured out. Further 30-minute check-ins
would cost the user credits to re-confirm saturation. The arms are stopped, the
champion is saved, everything is committed and the dashboard is published.

**To restart:** `/loop 30m <the same prompt>`, or launch an arm directly with
`./scripts/run_experiment.sh <name> --hours N ...` (see check-in 13 for flags).

**FINAL: 5 wins, 10 refutations, 1 machinery bug fixed, 1 self-correction.**
Champion Sharpe in the informed market: ~10 (uninformed policy dropped in) -> 34.8.

### Check-in 16 — structural change: **recursive estimator**, +2.54 (t=2.17)

Two structural candidates probed before building either.

**PROBE A — size skew on the estimate: REFUTED.** Variable quote size was
refuted in the uninformed market, but never retested where a directional signal
exists, and unlike position-taking it is free (same passive quotes, different
size shown). Measured (block 500k, n=48):
    k=0 (control) -0.86 | k=0.6 +0.63 | k=1.2 +0.22 | k=2.0 -2.28
All inside the control's own noise. The lean already expresses the view
efficiently; a second channel adds nothing.

**PROBE B — recursive estimator: BUILT.** Motivation is structural rather than
empirical: the gap is a latent random walk observed through noisy signed flow,
which is a linear-Gaussian state-space problem whose optimal estimator is
*recursive*. A lag stack truncates that recursion at 5 taps. On identical
on-policy data:
    lag-stack MLP  corr 0.547
    GRU hidden 48  corr 0.599
    GRU hidden 96  corr **0.608**
Built as `GRUGapNet`; the env carries one hidden vector per step and resets it
per episode, so it is O(1) state and needs no history buffer at all.
Trained on-policy: held-out corr **0.636**. In-env feature vs the true gap went
**0.42 -> 0.624**.

Screen (510k, n=64) over lean scales: 0.8x -0.11 | **1.0x +3.70** | 1.3x +3.24.
1.0x wins, so the promotion is the estimator ALONE -- cleanly isolated, as the
attribution rule from check-in 14 requires.
Confirm (520k, n=128): **+2.54, stderr 1.17, t=2.17, CI [+0.24,+4.85]**.
**PROMOTED.** Champion policy weights untouched; Sharpe 34.5 -> 37.0.

Seed blocks burned: ...500k 510k 520k. 50 tests passing.
**Session: 6 wins, 11 refutations.**
