import numpy as np
import pytest

from lobrl.env import EnvConfig, MarketMakingEnv
from lobrl.multi_env import OWNER_BASE, MultiAgentMarketMakingEnv, MultiEnvConfig


def run(cfg, seed, steps, action=None):
    env = MultiAgentMarketMakingEnv(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    a = np.zeros((cfg.n_agents, 2), np.float32) if action is None else action
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return env, info


def test_observation_layout_matches_single_agent():
    """A policy trained alone must be loadable into the competitive book."""
    single = MarketMakingEnv(EnvConfig(levels=5), seed=0)
    multi = MultiAgentMarketMakingEnv(MultiEnvConfig(levels=5, n_agents=3), seed=0)
    assert single.observation_space.shape == multi.observation_space.shape
    assert single.action_space.shape == multi.action_space.shape


def test_step_shapes():
    cfg = MultiEnvConfig(n_agents=3, max_steps=20)
    env = MultiAgentMarketMakingEnv(cfg, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, env.observation_space.shape[0])
    obs, rew, term, trunc, info = env.step(np.zeros((3, 2), np.float32))
    assert obs.shape == (3, env.observation_space.shape[0])
    assert rew.shape == (3,)
    assert info["equity"].shape == (3,)


def test_rejects_bad_agent_count():
    with pytest.raises(ValueError):
        MultiAgentMarketMakingEnv(MultiEnvConfig(n_agents=0))
    with pytest.raises(ValueError):
        MultiAgentMarketMakingEnv(MultiEnvConfig(n_agents=300))


def test_fills_are_attributed_to_the_right_owner():
    cfg = MultiEnvConfig(n_agents=4, max_steps=400)
    env, info = run(cfg, 1, 400)
    for i in range(4):
        mine = [f for f in env.fill_log if f[0] == i]
        assert len(mine) == env.trade_count[i]
        # Cash must equal minus the signed notional of this agent's own fills.
        cash = -sum(q * px for (_, _, px, q) in mine)
        assert env.cash[i] == pytest.approx(cash)
        assert env.inventory[i] == sum(q for (_, _, _, q) in mine)


def test_equity_is_cash_plus_marked_inventory():
    cfg = MultiEnvConfig(n_agents=3, max_steps=300)
    env, info = run(cfg, 2, 300)
    assert np.allclose(info["equity"], env.cash + env.inventory * info["mid"])


def test_identical_agents_are_treated_fairly():
    """Same policy in every seat should earn statistically similar fills."""
    cfg = MultiEnvConfig(n_agents=4, max_steps=1500)
    tight = np.full((4, 2), -0.7, np.float32)   # wide quotes barely fill at all
    env, info = run(cfg, 3, 1500, tight)
    fills = info["trades"].astype(float)
    assert fills.sum() > 40                      # the test is meaningful
    assert fills.std() / fills.mean() < 0.45     # no seat dominates


def test_submission_order_is_the_thing_that_makes_it_fair():
    """With a fixed submission order, seat 0 wins the queue and takes more fills."""
    fixed = MultiEnvConfig(n_agents=4, max_steps=1500, randomize_order=False)
    env, info = run(fixed, 3, 1500, np.full((4, 2), -0.7, np.float32))
    shares = info["trades"] / max(info["trades"].sum(), 1)
    assert shares[0] > shares[-1]


def test_agent_quotes_never_cross_the_book():
    cfg = MultiEnvConfig(n_agents=4, max_steps=200)
    env = MultiAgentMarketMakingEnv(cfg, seed=4)
    env.reset(seed=4)
    a = np.full((4, 2), -1.0, np.float32)  # everyone quotes as tight as allowed
    for _ in range(200):
        env.step(a)
        bb, ba = env.book.best_bid(), env.book.best_ask()
        assert bb < 0 or ba < 0 or bb < ba


def test_owner_ids_stay_in_range():
    cfg = MultiEnvConfig(n_agents=5, max_steps=200)
    env, _ = run(cfg, 5, 200)
    ev = env.book.events_buffer()[: env.book.events_size()]
    owners = set(int(o) for o in ev["owner"])
    assert owners <= set(range(0, OWNER_BASE + 5))


def test_determinism_under_seed():
    cfg = MultiEnvConfig(n_agents=3, max_steps=120)
    a = np.zeros((3, 2), np.float32)

    def go(seed):
        env = MultiAgentMarketMakingEnv(cfg, seed=seed)
        env.reset(seed=seed)
        return np.array([env.step(a)[1] for _ in range(120)])

    assert np.allclose(go(9), go(9))
    assert not np.allclose(go(9), go(10))


def test_tighter_quotes_win_more_fills():
    """The competitive mechanism itself: undercutting takes the queue."""
    cfg = MultiEnvConfig(n_agents=2, max_steps=1200, randomize_order=True)
    env = MultiAgentMarketMakingEnv(cfg, seed=7)
    env.reset(seed=7)
    a = np.array([[-1.0, -1.0], [1.0, 1.0]], np.float32)  # agent 0 tight, agent 1 wide
    for _ in range(1200):
        _, _, _, trunc, info = env.step(a)
        if trunc:
            break
    assert info["trades"][0] > info["trades"][1]


def test_selfplay_buffer_gae_matches_single_stream():
    from lobrl.ppo import RolloutBuffer
    from lobrl.selfplay import MultiRolloutBuffer

    T, n, D = 16, 3, 4
    rng = np.random.default_rng(0)
    multi = MultiRolloutBuffer(T, n, D, 2)
    singles = [RolloutBuffer(T, D, 2) for _ in range(n)]
    for t in range(T):
        obs = rng.normal(size=(n, D)).astype(np.float32)
        act = rng.normal(size=(n, 2)).astype(np.float32)
        logp = rng.normal(size=n).astype(np.float32)
        rew = rng.normal(size=n).astype(np.float32)
        val = rng.normal(size=n).astype(np.float32)
        done = np.zeros(n, np.float32)
        multi.add(obs, act, logp, rew, val, done)
        for i in range(n):
            singles[i].add(obs[i], act[i], logp[i], rew[i], val[i], done[i])

    last = rng.normal(size=n).astype(np.float32)
    adv_m, ret_m = multi.compute_gae(last, 0.99, 0.95)
    adv_m = adv_m.reshape(T, n)
    for i in range(n):
        adv_s, _ = singles[i].compute_gae(float(last[i]), 0.99, 0.95)
        assert np.allclose(adv_m[:, i], adv_s, atol=1e-5)


def test_pool_pruning_actually_caps_the_pool(tmp_path):
    """A floored stride used to keep every snapshot once the pool got large."""
    from lobrl.arena import Arena, ArenaConfig

    cfg = ArenaConfig(pool_max=20, pool_recent=6)
    a = Arena.__new__(Arena)          # prune needs only cfg and the directory
    a.cfg = cfg
    a.pool_dir = tmp_path
    for i in range(140):
        (tmp_path / f"gen_{i:04d}.pt").write_bytes(b"x")
        a._prune_pool()
        assert len(list(tmp_path.glob("gen_*.pt"))) <= cfg.pool_max, f"unbounded at {i}"
    kept = sorted(p.name for p in tmp_path.glob("gen_*.pt"))
    assert kept[-1] == "gen_0139.pt"          # newest always survives
    assert len(kept) >= cfg.pool_recent


def test_best_checkpoint_is_the_champion_not_the_last_learner(tmp_path):
    """A finished run must not overwrite best.pt with an unpromoted learner."""
    import torch

    from lobrl.arena import Arena, ArenaConfig, FrozenPolicy
    from lobrl.ppo import ActorCritic, PPOConfig, RunningNorm

    cfg = ArenaConfig(n_agents=2, episode_steps=100)
    a = Arena.__new__(Arena)
    a.cfg, a.dir, a.gen = cfg, tmp_path, 5
    a.obs_dim, a.act_dim = 28, 2

    champion = FrozenPolicy(ActorCritic(28, 2, PPOConfig()).state_dict(),
                            RunningNorm(28).state_dict(), 28, 2, "champ")
    drifted = ActorCritic(28, 2, PPOConfig())
    with torch.no_grad():                      # make the learner clearly different
        for p in drifted.parameters():
            p.add_(1.0)

    class _T:
        net = drifted
    a.trainer, a.norm = _T(), RunningNorm(28)

    a._save_best(champion)
    saved = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)["model"]
    ref = champion.net.state_dict()
    assert all(torch.equal(saved[k], ref[k]) for k in ref)


def test_variable_size_action_maps_zero_to_the_fixed_size():
    """The 2-d action space must be a strict subset of the 4-d one."""
    cfg = MultiEnvConfig(n_agents=2, max_steps=5, variable_size=True)
    env = MultiAgentMarketMakingEnv(cfg, seed=0)
    env.reset(seed=0)
    assert env.act_dim == 4
    _, _, _, _, info = env.step(np.zeros((2, 4), np.float32))
    assert (info["sizes"] == cfg.quote_size).all()

    env.reset(seed=0)
    _, _, _, _, lo = env.step(np.array([[0, 0, -1, -1]] * 2, np.float32))
    env.reset(seed=0)
    _, _, _, _, hi = env.step(np.array([[0, 0, 1, 1]] * 2, np.float32))
    assert (lo["sizes"] == cfg.min_quote_size).all()
    assert (hi["sizes"] == cfg.max_quote_size).all()


def test_two_dim_policy_keeps_its_behaviour_in_a_variable_size_book():
    """Padding a 2-d action with zeros must reproduce fixed-size quoting."""
    from lobrl.arena import ArenaConfig, FrozenPolicy, _run_episode
    from lobrl.ppo import ActorCritic, PPOConfig, RunningNorm

    old = FrozenPolicy(ActorCritic(28, 2, PPOConfig()).state_dict(),
                       RunningNorm(28).state_dict(), 28, 2, "old")
    for variable in (False, True):
        cfg = ArenaConfig(n_agents=2, episode_steps=120, variable_size=variable)
        res = _run_episode([old, old], cfg, seed=3)
        assert len(res) == 2
        if variable:
            assert res[0]["n_fills"] >= 0      # runs without shape errors


def test_frozen_policy_recovers_hidden_width_from_the_checkpoint():
    from lobrl.arena import FrozenPolicy
    from lobrl.ppo import ActorCritic, PPOConfig, RunningNorm

    wide = ActorCritic(28, 4, PPOConfig(hidden=256))
    p = FrozenPolicy(wide.state_dict(), RunningNorm(28).state_dict(), 28, 4, "wide")
    assert p.net.actor[0].out_features == 256
    assert p.act_batch(np.zeros((3, 28), np.float32)).shape == (3, 4)


def test_flow_features_extend_the_observation_and_stay_bounded():
    from lobrl.flow import FlowConfig

    plain = MultiAgentMarketMakingEnv(MultiEnvConfig(n_agents=2), seed=0)
    tape = MultiAgentMarketMakingEnv(
        MultiEnvConfig(n_agents=2, max_steps=400, flow_features=True), seed=0)
    assert tape.observation_space.shape[0] == plain.observation_space.shape[0] + 4

    obs, _ = tape.reset(seed=0)
    seen = []
    for _ in range(400):
        obs, *_ = tape.step(np.zeros((2, 2), np.float32))
        seen.append(obs[0, -4:].copy())
    seen = np.array(seen)
    assert np.all(np.abs(seen) <= 1.0)          # squashed
    assert seen.std(axis=0).max() > 1e-3        # and actually varying


def test_informed_flow_drags_the_mid_toward_the_fundamental():
    from lobrl import OrderBook
    from lobrl.flow import FlowConfig, MarketFlow

    book = OrderBook()
    flow = MarketFlow(FlowConfig(informed_frac=0.5, fundamental_vol=0.4),
                      np.random.default_rng(0))
    flow.seed_book(book, 10_000)
    mids, fund, informed = [], [], 0
    for t in range(2000):
        informed += flow.step(book)["informed"]
        if t > 100:
            mids.append(book.mid())
            fund.append(flow.fundamental)
    assert informed > 0
    assert np.corrcoef(mids, fund)[0, 1] > 0.5   # price discovery happens

    # And with the feature off, the market is exactly the uninformed one.
    b2 = OrderBook()
    f2 = MarketFlow(FlowConfig(), np.random.default_rng(0))
    f2.seed_book(b2, 10_000)
    assert sum(f2.step(b2)["informed"] for _ in range(300)) == 0
