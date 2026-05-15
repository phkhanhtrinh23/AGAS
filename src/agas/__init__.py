"""AGAS — Agentic Group Attack System for Recommender Systems.

The public API consists of three paper-aligned modules:

* :mod:`agas.roles`      – the four worker roles ``{PR, SN, CA, IN}``.
* :mod:`agas.signals`    – worker signals ``τ, γ, φ`` and environment signals
                           ``ρ, Δρ, η, ξ, a``.
* :mod:`agas.strategies` – the eight Coordinator strategies and the rule-based
                           selector that materialises the triggers from
                           ``method_strategies.tex``.
"""

from agas.roles import Role, ROLE_LONG_NAMES
from agas.signals import (
    EnvSignals,
    SuppressionSignal,
    WorkerSignals,
    initial_env_signals,
    initial_worker_signals,
)
from agas.strategies import (
    STRATEGY_DESCRIPTIONS,
    Strategy,
    StrategySelectorConfig,
    select_strategy,
)

__all__ = [
    "__version__",
    "Role",
    "ROLE_LONG_NAMES",
    "WorkerSignals",
    "EnvSignals",
    "SuppressionSignal",
    "initial_worker_signals",
    "initial_env_signals",
    "Strategy",
    "STRATEGY_DESCRIPTIONS",
    "StrategySelectorConfig",
    "select_strategy",
]
__version__ = "0.1.0"
