"""Tests for the eight-strategy enum and the rule-based selector."""

from __future__ import annotations

from agas.roles import Role
from agas.signals import EnvSignals, SuppressionSignal, WorkerSignals
from agas.strategies import (
    STRATEGY_DESCRIPTIONS,
    Strategy,
    StrategySelectorConfig,
    select_strategy,
)


def test_eight_strategies_present_and_documented() -> None:
    """The enum must contain exactly the eight paper strategies, and every
    member must have a description string."""

    members = Strategy.all()
    assert len(members) == 8

    names = {s.name for s in members}
    assert names == {
        "S1_VICTIM_PROBE",
        "S2_BRIDGE_BUILDING",
        "S3_WARM_UP",
        "S4_FIRST_PUSH",
        "S5_SILENT_SLOWDOWN",
        "S6_PROFILE_CLEANUP",
        "S7_SAFE_REPLACEMENT",
        "S8_MAIN_ATTACK",
    }

    for strat in members:
        assert strat in STRATEGY_DESCRIPTIONS
        assert isinstance(STRATEGY_DESCRIPTIONS[strat], str)
        assert STRATEGY_DESCRIPTIONS[strat].strip() != ""


def test_selector_returns_s1_in_probe_window() -> None:
    """During the probe window (t < probe_rounds) S1 fires unconditionally."""

    env = EnvSignals(rho=100.0, delta_rho=0.0, eta=1.0, xi=SuppressionSignal(), alert=0)
    workers = {"a": WorkerSignals(), "b": WorkerSignals()}
    strat = select_strategy(round_index=0, env_signals=env, worker_signals=workers)
    assert strat == Strategy.S1_VICTIM_PROBE


def test_selector_returns_s8_in_default_state() -> None:
    """After the probe / warm-up window with no alert signals, the default is
    the Main Attack."""

    env = EnvSignals(rho=10.0, delta_rho=1.0, eta=1.0, xi=SuppressionSignal(), alert=0)
    workers = {"a": WorkerSignals(trust=2.0, risk=0.0, validator=0.0)}
    cfg = StrategySelectorConfig(probe_rounds=2, warmup_rounds=2, first_push_round=4)
    # round 5 → past probe window, past warm-up, past first push.
    strat = select_strategy(round_index=5, env_signals=env, worker_signals=workers, config=cfg)
    assert strat == Strategy.S8_MAIN_ATTACK


def test_selector_returns_s5_on_silent_slowdown() -> None:
    """S5 fires when η < eta_low AND alert == 0."""

    env = EnvSignals(rho=10.0, delta_rho=0.0, eta=0.3, xi=SuppressionSignal(), alert=0)
    workers = {"a": WorkerSignals()}
    cfg = StrategySelectorConfig()
    strat = select_strategy(round_index=5, env_signals=env, worker_signals=workers, config=cfg)
    assert strat == Strategy.S5_SILENT_SLOWDOWN


def test_selector_returns_s6_on_high_validator() -> None:
    """S6 fires when any worker's φ exceeds phi_high."""

    env = EnvSignals(rho=10.0, delta_rho=0.0, eta=1.0, xi=SuppressionSignal(q=0.0, s=0), alert=0)
    workers = {"a": WorkerSignals(validator=0.9)}
    strat = select_strategy(round_index=5, env_signals=env, worker_signals=workers)
    assert strat == Strategy.S6_PROFILE_CLEANUP


def test_selector_returns_s7_on_alert_plus_high_q() -> None:
    """S7 fires when a_t == 1 AND q_t >= q_high."""

    env = EnvSignals(rho=10.0, delta_rho=0.0, eta=1.0, xi=SuppressionSignal(q=0.7, s=0), alert=1)
    workers = {"a": WorkerSignals()}
    strat = select_strategy(round_index=5, env_signals=env, worker_signals=workers)
    assert strat == Strategy.S7_SAFE_REPLACEMENT


def test_selector_returns_s2_when_victim_is_graph_and_pool_empty() -> None:
    """S2 fires for graph victims while the bridge pool is still empty."""

    env = EnvSignals(rho=10.0, delta_rho=0.0, eta=1.0, xi=SuppressionSignal(), alert=0)
    workers = {"a": WorkerSignals()}
    strat = select_strategy(
        round_index=2,
        env_signals=env,
        worker_signals=workers,
        victim_is_graph=True,
        bridge_pool_size=0,
    )
    assert strat == Strategy.S2_BRIDGE_BUILDING


def test_roles_enum_contains_paper_symbols() -> None:
    """The role enum must use the paper's short symbols PR/SN/CA/IN."""

    names = {r.value for r in Role.all()}
    assert names == {"PR", "SN", "CA", "IN"}
