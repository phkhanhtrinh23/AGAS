"""The eight Coordinator strategies from ``method_strategies.tex``.

Each round the Coordinator picks exactly one strategy from the enum below.
Triggers are the plain-English rules listed in the paper, implemented as a
small rule-based selector that the LLM Coordinator can override (the LLM
policy still has to pick a value from this enum).

Strategy list (paper labels)
---------------------------
1. ``S1_VICTIM_PROBE``        – first few rounds: probe the victim family.
2. ``S2_BRIDGE_BUILDING``     – graph-only: build a bridge-item pool.
3. ``S3_WARM_UP``             – build worker trust before snipers fire.
4. ``S4_FIRST_PUSH``          – first coordinated sniper payload.
5. ``S5_SILENT_SLOWDOWN``     – ``η_t`` low *and* ``a_t = 0``.
6. ``S6_PROFILE_CLEANUP``     – high ``φ`` / large ``q_t`` / ``a_t = 1``.
7. ``S7_SAFE_REPLACEMENT``    – high-risk state, swap in rested workers.
8. ``S8_MAIN_ATTACK``         – no strong suspicious signals, push the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Strategy(str, Enum):
    """The eight AGAS round-level strategies (see ``method_strategies.tex``)."""

    S1_VICTIM_PROBE = "S1_VICTIM_PROBE"
    S2_BRIDGE_BUILDING = "S2_BRIDGE_BUILDING"
    S3_WARM_UP = "S3_WARM_UP"
    S4_FIRST_PUSH = "S4_FIRST_PUSH"
    S5_SILENT_SLOWDOWN = "S5_SILENT_SLOWDOWN"
    S6_PROFILE_CLEANUP = "S6_PROFILE_CLEANUP"
    S7_SAFE_REPLACEMENT = "S7_SAFE_REPLACEMENT"
    S8_MAIN_ATTACK = "S8_MAIN_ATTACK"

    @classmethod
    def all(cls) -> list["Strategy"]:
        """Return the canonical paper ordering ``[S1, ..., S8]``."""

        return [
            cls.S1_VICTIM_PROBE,
            cls.S2_BRIDGE_BUILDING,
            cls.S3_WARM_UP,
            cls.S4_FIRST_PUSH,
            cls.S5_SILENT_SLOWDOWN,
            cls.S6_PROFILE_CLEANUP,
            cls.S7_SAFE_REPLACEMENT,
            cls.S8_MAIN_ATTACK,
        ]


STRATEGY_DESCRIPTIONS: dict[Strategy, str] = {
    Strategy.S1_VICTIM_PROBE: (
        "Victim Probe: in the first few rounds, assign Profilers to probe the "
        "victim family. A favourable Δρ^{(t)} on direct target probes => victim "
        "is treated as embedding-based, otherwise graph-based."
    ),
    Strategy.S2_BRIDGE_BUILDING: (
        "Bridge Building: graph-only. Profilers rank non-target items by how "
        "often real users connected to the target rate them, then commit those "
        "items into the bridge-item pool used by Snipers."
    ),
    Strategy.S3_WARM_UP: (
        "Warm-up: after probing, build worker trust τ_{t,w} via one Profiler + "
        "a few Camouflageurs before any Sniper fires."
    ),
    Strategy.S4_FIRST_PUSH: (
        "First Push: launch the first coordinated payload. A small number of "
        "Snipers attack while >=1 Camouflageur stays active as cover."
    ),
    Strategy.S5_SILENT_SLOWDOWN: (
        "Silent Slowdown: when η_t drops below the early baseline but the "
        "alert flag a_t = 0, reduce pressure — let high-trust workers behave "
        "as Camouflageurs for a few rounds."
    ),
    Strategy.S6_PROFILE_CLEANUP: (
        "Profile Cleanup: when validator φ_{t,w} is high for some workers, or "
        "ξ_t = (q_t, s_t) climbs, or a_t = 1: pause suspicious Snipers, lock "
        "high-risk workers, avoid heavy-overlap profiles."
    ),
    Strategy.S7_SAFE_REPLACEMENT: (
        "Safe Replacement: high-risk state (a_t = 1 or large q_t). Cool down "
        "the riskiest workers and rotate in rested, lower-risk workers in "
        "safer roles."
    ),
    Strategy.S8_MAIN_ATTACK: (
        "Main Attack: no strong suspicious signals and the target is still "
        "far from top-K. Promote the best available worker to Sniper and keep "
        "pushing."
    ),
}


@dataclass
class StrategySelectorConfig:
    """Threshold parameters for the rule-based strategy selector.

    These thresholds materialise the plain-English triggers from
    ``method_strategies.tex``. They are exposed as a dataclass so an ablation
    can sweep them.
    """

    probe_rounds: int = 2
    warmup_rounds: int = 2
    first_push_round: int = 4
    eta_low: float = 0.6
    q_high: float = 0.5
    phi_high: float = 0.6
    bridge_round_window: tuple[int, int] = (1, 3)  # rounds during which S2 may fire


def select_strategy(
    *,
    round_index: int,
    env_signals,  # signals.EnvSignals
    worker_signals: dict,  # mapping worker_id -> signals.WorkerSignals
    victim_is_graph: Optional[bool] = None,
    config: Optional[StrategySelectorConfig] = None,
    bridge_pool_size: int = 0,
) -> Strategy:
    """Rule-based selector returning one of the eight paper strategies.

    The rules below are written so a reviewer can audit them against
    ``method_strategies.tex`` line by line. Priority follows the paper:

    1. ``S1`` for the first few probe rounds.
    2. ``S6`` whenever ``φ_{t,w}`` is high for any worker, or ``a_t = 1``,
       or ``q_t`` is high.
    3. ``S7`` for explicit alert states (``a_t = 1`` with large ``q_t``).
    4. ``S5`` for silent suppression: low ``η_t`` and ``a_t = 0``.
    5. ``S2`` when the victim is graph-style and the bridge pool is empty.
    6. ``S3`` during the warm-up window.
    7. ``S4`` for the first push round.
    8. ``S8`` otherwise — the main-attack default.

    Args:
        round_index: Zero-indexed round ``t``.
        env_signals: Current ``EnvSignals`` (``rho``, ``delta_rho``, ``eta``,
            ``xi=(q,s)``, ``alert``).
        worker_signals: Mapping ``w -> WorkerSignals`` for each worker.
        victim_is_graph: Optional probe outcome from S1. When ``True`` the
            victim is treated as graph-style, unlocking S2.
        config: Optional selector thresholds.
        bridge_pool_size: How many bridge items have already been collected.

    Returns:
        The chosen :class:`Strategy`.
    """

    cfg = config or StrategySelectorConfig()
    eta = float(getattr(env_signals, "eta", 1.0))
    q = float(getattr(env_signals, "q", 0.0))
    alert = int(getattr(env_signals, "alert", 0))
    phi_max = max((float(s.validator) for s in worker_signals.values()), default=0.0)

    # 1. Victim Probe — first few rounds.
    if round_index < cfg.probe_rounds:
        return Strategy.S1_VICTIM_PROBE

    # 2/3. Profile cleanup vs Safe replacement: both react to high suspicion,
    # but Safe Replacement is the harsher response when the alert *and* q_t
    # are both above threshold.
    if alert == 1 and q >= cfg.q_high:
        return Strategy.S7_SAFE_REPLACEMENT
    if phi_max >= cfg.phi_high or q >= cfg.q_high or alert == 1:
        return Strategy.S6_PROFILE_CLEANUP

    # 4. Silent Slowdown — low η, no alert.
    if eta < cfg.eta_low and alert == 0:
        return Strategy.S5_SILENT_SLOWDOWN

    # 5. Bridge Building — only relevant for graph-style victims.
    if (
        victim_is_graph
        and bridge_pool_size == 0
        and cfg.bridge_round_window[0] <= round_index <= cfg.bridge_round_window[1]
    ):
        return Strategy.S2_BRIDGE_BUILDING

    # 6. Warm-up – right after the probe window.
    if round_index < cfg.probe_rounds + cfg.warmup_rounds:
        return Strategy.S3_WARM_UP

    # 7. First Push — exactly one round.
    if round_index == cfg.first_push_round:
        return Strategy.S4_FIRST_PUSH

    # 8. Default: Main Attack.
    return Strategy.S8_MAIN_ATTACK


__all__ = [
    "Strategy",
    "STRATEGY_DESCRIPTIONS",
    "StrategySelectorConfig",
    "select_strategy",
]
