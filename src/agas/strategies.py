"""The eight Coordinator strategies from ``method_strategies.tex``."""

from __future__ import annotations

from enum import Enum


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


__all__ = [
    "Strategy",
    "STRATEGY_DESCRIPTIONS",
]
