import numpy as np
import pytest

from lobrl.baselines import AvellanedaStoikovPolicy, FixedSpreadPolicy
from lobrl.env import EnvConfig, MarketMakingEnv
from lobrl.flow import FlowConfig, MarketFlow
from lobrl.metrics import adverse_selection, max_drawdown, sharpe
from lobrl import OrderBook, Side


def test_env_contract():
    env = MarketMakingEnv(EnvConfig(max_steps=50), seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert np.isfinite(r)
    assert set(info) >= {"mid", "inventory", "cash", "equity"}


def test_episode_truncates_at_max_steps():
    env = MarketMakingEnv(EnvConfig(max_steps=25), seed=1)
    env.reset(seed=1)
    for i in range(25):
        _, _, term, trunc, _ = env.step(np.zeros(2, np.float32))
    assert trunc and not term


def test_determinism_under_seed():
    def run(seed):
        env = MarketMakingEnv(EnvConfig(max_steps=60), seed=seed)
        env.reset(seed=seed)
        rs = [env.step(np.zeros(2, np.float32))[1] for _ in range(60)]
        return np.array(rs)

    assert np.allclose(run(7), run(7))
    assert not np.allclose(run(7), run(8))


def test_cash_and_inventory_are_consistent():
    env = MarketMakingEnv(EnvConfig(max_steps=200), seed=2)
    env.reset(seed=2)
    for _ in range(200):
        _, _, _, _, info = env.step(np.array([-0.5, -0.5], np.float32))
    assert info["equity"] == pytest.approx(info["cash"] + info["inventory"] * info["mid"])


def test_agent_quotes_never_cross_the_book():
    env = MarketMakingEnv(EnvConfig(max_steps=100), seed=3)
    env.reset(seed=3)
    for _ in range(100):
        env.step(np.array([-1.0, -1.0], np.float32))  # tightest allowed quotes
        bb, ba = env.book.best_bid(), env.book.best_ask()
        assert bb < 0 or ba < 0 or bb < ba


def test_inventory_cap_stops_one_sided_quoting():
    cfg = EnvConfig(max_steps=400, max_inventory=40)
    env = MarketMakingEnv(cfg, seed=4)
    env.reset(seed=4)
    for _ in range(400):
        _, _, _, _, info = env.step(np.array([-1.0, 1.0], np.float32))
    # Quoting stops on the side that would breach, so the cap is only ever
    # exceeded by at most one fill's worth of size.
    assert abs(info["inventory"]) <= cfg.max_inventory + cfg.quote_size


def test_flow_keeps_a_two_sided_book():
    book = OrderBook()
    flow = MarketFlow(FlowConfig(), np.random.default_rng(0))
    flow.seed_book(book, 10_000)
    spreads = []
    for t in range(500):
        flow.step(book)
        if t > 50:
            spreads.append(book.spread())
    assert min(spreads) >= 1
    assert np.mean(spreads) < 20        # book refills after sweeps
    assert book.rejects() == 0


def test_flow_produces_clustered_trades():
    book = OrderBook()
    flow = MarketFlow(FlowConfig(), np.random.default_rng(1))
    flow.seed_book(book, 10_000)
    counts = np.array([flow.step(book)["markets"] for _ in range(1000)])
    # Hawkes excitation makes trade counts over-dispersed relative to Poisson.
    assert counts.var() > counts.mean()


def test_baselines_emit_valid_actions():
    env = MarketMakingEnv(EnvConfig(max_steps=30), seed=5)
    obs, _ = env.reset(seed=5)
    for pol in (FixedSpreadPolicy(3.0), AvellanedaStoikovPolicy()):
        for _ in range(30):
            a = pol(obs, env)
            assert env.action_space.contains(np.asarray(a, np.float32))
            obs, *_ = env.step(a)


def test_metrics():
    assert max_drawdown(np.array([0.0, 5.0, 1.0, 4.0])) == 4.0
    assert sharpe(np.zeros(10)) == 0.0
    assert sharpe(np.ones(10)) == 0.0        # zero variance -> undefined, reported as 0
    mids = np.array([100.0] * 5 + [90.0] * 20)
    # bought at 100, mid falls to 90 ten steps later -> adversely selected by 10
    assert adverse_selection([(0, 100.0, 10)], mids, 10) == 10.0
