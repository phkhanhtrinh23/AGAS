"""Hand-checked tests for the paper-aligned worker / environment signals.

These tests pin the update equations from ``method_coordinator.tex`` so that
``code → paper`` traceability cannot regress silently.
"""

from __future__ import annotations

import pytest

from agas.signals import (
    EnvSignals,
    SuppressionSignal,
    WorkerSignals,
    initial_env_signals,
    initial_worker_signals,
)


# ---------------------------------------------------------------------------
# eq:trust_update + eq:risk_update — hand-checked tiny scenario
# ---------------------------------------------------------------------------


def test_trust_increments_only_when_deviation_within_one() -> None:
    """eq:trust_update — τ goes up exactly when |r - b| ≤ 1."""

    sig = WorkerSignals()
    sig.update_after_action(rating=4.0, item_bias=3.5)  # d = 0.5
    assert sig.trust == 1.0
    assert sig.risk == 0.0  # no extreme deviation

    sig.update_after_action(rating=5.0, item_bias=3.0)  # d = 2.0
    # τ should *not* increase because d > 1
    assert sig.trust == 1.0
    # risk should add 0.5 (d > 1) + 0.5 (d > 1.5 AND τ_before < 1 is False here)
    # τ_before = 1.0 so the second 0.5 does NOT fire.
    assert sig.risk == pytest.approx(0.5)


def test_risk_extreme_plus_low_trust_adds_full_unit() -> None:
    """eq:risk_update — d > 1.5 *and* τ < 1 adds +0.5 on top of the base +0.5."""

    sig = WorkerSignals()
    sig.update_after_action(rating=1.0, item_bias=3.0)  # d = 2.0, τ_before = 0
    # base 0.5 + low-trust 0.5 = 1.0
    assert sig.risk == pytest.approx(1.0)
    assert sig.trust == 0.0  # τ stays at 0 (d > 1)


def test_risk_decay_on_inactivity() -> None:
    """eq:risk_decay — inactivity removes 0.5 risk units, clamped at 0."""

    sig = WorkerSignals(risk=0.6)
    sig.apply_inactive_decay()
    assert sig.risk == pytest.approx(0.1)
    sig.apply_inactive_decay()
    assert sig.risk == 0.0  # clamped at 0


def test_alert_or_block_adds_extra_risk_penalty() -> None:
    """An explicit alert / block raises an extra 0.5 risk penalty."""

    sig = WorkerSignals()
    sig.update_after_action(rating=3.5, item_bias=3.5, alert_raised=True)
    # d = 0 → trust +1, risk = 0 from rules + 0.5 from alert.
    assert sig.trust == 1.0
    assert sig.risk == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# eq:profile_validator — φ_{t,w}
# ---------------------------------------------------------------------------


def test_profile_validator_uses_paper_weights() -> None:
    """φ = min(1, 0.4 e + 0.3 h + 0.3 o)."""

    sig = WorkerSignals()
    sig.update_validator(extreme_ratio=0.5, target_positive=True, overlap_fraction=0.0)
    # 0.4 * 0.5 + 0.3 * 1 + 0.3 * 0 = 0.5
    assert sig.validator == pytest.approx(0.5)

    sig.update_validator(extreme_ratio=1.0, target_positive=True, overlap_fraction=1.0)
    # 0.4 + 0.3 + 0.3 = 1.0
    assert sig.validator == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Environment signals — eq:round_suppression_terms + eq:round_suppression_score
# ---------------------------------------------------------------------------


def test_env_signal_initialisation_matches_alg_line_9() -> None:
    """Initial values from algorithms/agas_end_to_end.tex line 9."""

    env = initial_env_signals(rho_zero=100.0)
    assert env.rho == 100.0
    assert env.delta_rho == 0.0
    assert env.eta == 1.0
    assert env.xi.to_tuple() == (0.0, 0)
    assert env.alert == 0


def test_env_signal_q_t_is_02_weighted_sum() -> None:
    """q_t = 0.2*(d̂ + δ̂ + m̂ + ŝ + g).

    Use a hand-constructed scenario: 5 attempts / 1 dropped / 4 accepted, all
    target actions weakly moving (m̂ = 1), no discounts, no overlap.
    """

    env = initial_env_signals(rho_zero=100.0)
    env.update(
        prev_rho=100.0,
        new_rho=100.0,
        attempts=5,
        accepted=4,
        dropped=1,
        target_actions=2,
        discount_magnitudes=[0.0, 0.0, 0.0, 0.0],
        weak_target_movements=2,
        worker_item_sets={"a": ["1", "2"], "b": ["3"]},  # disjoint → g = 0
        spike_alert=False,
        sharp_acceptance_drop=False,
    )

    # d̂ = 1/5 = 0.2; δ̂ = 0; m̂ = 2/2 = 1; this is the first "bad" round so s
    # bumps to 1 → ŝ = 1/3; g = 0.
    expected = 0.2 * 0.2 + 0.2 * 0.0 + 0.2 * 1.0 + 0.2 * (1.0 / 3.0) + 0.2 * 0.0
    assert env.q == pytest.approx(expected, abs=1e-6)
    assert env.delta_rho == 0.0
    assert env.eta == pytest.approx(4 / 5)
    assert env.alert == 0  # no spike, no sharp drop


def test_env_signal_alert_fires_on_spike() -> None:
    """a_t = 1 when a spike or sharp acceptance drop is reported."""

    env = initial_env_signals(rho_zero=10.0)
    env.update(
        prev_rho=10.0,
        new_rho=10.0,
        attempts=1,
        accepted=1,
        dropped=0,
        target_actions=1,
        discount_magnitudes=[0.0],
        weak_target_movements=0,
        worker_item_sets={"a": ["x"]},
        spike_alert=True,
        sharp_acceptance_drop=False,
    )
    assert env.alert == 1


def test_initial_worker_signals_factory() -> None:
    """All τ, γ, φ start at 0 (algorithms/agas_end_to_end.tex line 9)."""

    ws = initial_worker_signals(["a", "b", "c"])
    assert set(ws) == {"a", "b", "c"}
    for sig in ws.values():
        assert sig.trust == 0.0
        assert sig.risk == 0.0
        assert sig.validator == 0.0


def test_suppression_signal_to_dict() -> None:
    s = SuppressionSignal(q=0.4, s=2)
    assert s.to_dict() == {"q": 0.4, "s": 2}
