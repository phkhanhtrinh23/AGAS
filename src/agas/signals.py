"""Worker and environment signals for AGAS.

This module is the paper-faithful implementation of the control signals
introduced in ``method_coordinator.tex``. Names and update equations follow
the manuscript exactly so that ``code -> paper`` traceability is direct.

Equation references
-------------------
* ``WorkerSignals.update_after_action``  → ``\\eqref{eq:trust_update}``,
                                          ``\\eqref{eq:risk_update}``.
* ``WorkerSignals.apply_inactive_decay`` → ``\\eqref{eq:risk_decay}``.
* ``WorkerSignals.update_validator``     → ``\\eqref{eq:profile_validator}``.
* ``EnvSignals.update``                  → ``\\eqref{eq:round_suppression_terms}``,
                                          ``\\eqref{eq:round_group_overlap}``,
                                          ``\\eqref{eq:round_suppression_score}``.

The signals are deliberately implemented as small dataclasses with pure
methods so that they can be unit-tested by hand against the equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Worker signals — τ_{t,w}, γ_{t,w}, φ_{t,w}
# ---------------------------------------------------------------------------


@dataclass
class WorkerSignals:
    """Per-worker signals tracked across rounds.

    Fields
    ------
    trust:
        ``τ_{t,w}`` – cumulative trust score. Increases by one only when the
        accepted rating is within one point of the current item bias
        (eq. :math:`\\tau_{t+1,w}=\\max(0,\\tau_{t,w}+\\mathbb{I}[d_{t,w}\\le 1])`).
    risk:
        ``γ_{t,w}`` – cumulative risk score. Increases by 0.5 when the rating
        deviates by more than 1 from the item bias, plus another 0.5 when the
        deviation exceeds 1.5 *and* ``τ_{t,w} < 1`` (extreme + low-trust).
        Decays by 0.5 per inactive round (``eq:risk_decay``). An extra penalty
        is added when an explicit alert is raised on the worker.
    validator:
        ``φ_{t,w}`` – internal validator score combining the extremity ratio
        ``e_{t,w}``, the target-positive flag ``h_{t,w}``, and the fake-user
        overlap fraction ``o_{t,w}`` via
        :math:`\\phi=\\min(1,0.4 e + 0.3 h + 0.3 o)`.
    """

    trust: float = 0.0
    risk: float = 0.0
    validator: float = 0.0

    # ------------------------------------------------------------------
    # τ and γ updates (eq:trust_update, eq:risk_update)
    # ------------------------------------------------------------------
    def update_after_action(
        self,
        rating: float,
        item_bias: float,
        *,
        alert_raised: bool = False,
        blocked: bool = False,
    ) -> None:
        """Apply :math:`\\tau` and :math:`\\gamma` updates after one accepted action.

        Args:
            rating: Worker-submitted rating value ``r_{t,w}``.
            item_bias: Current item bias ``b_i`` against which the rating is
                compared. The deviation ``d_{t,w}=|r_{t,w}-b_i|`` controls both
                updates per ``method_coordinator.tex``.
            alert_raised: ``True`` when AGAS raised an explicit alert on this
                worker in the current round. Adds an extra risk penalty
                (paper: *"If the worker is blocked, or an explicit alert is
                raised, AGAS adds an extra risk penalty"*).
            blocked: ``True`` when the action was blocked by the platform.
                Also triggers the extra risk penalty.
        """

        # Trust τ_{t,w} captured *before* the update is used in the γ rule.
        tau_before = self.trust
        d = abs(float(rating) - float(item_bias))

        # eq:trust_update
        if d <= 1.0:
            self.trust = max(0.0, tau_before + 1.0)
        else:
            self.trust = max(0.0, tau_before)

        # eq:risk_update
        risk_delta = 0.0
        if d > 1.0:
            risk_delta += 0.5
        if d > 1.5 and tau_before < 1.0:
            risk_delta += 0.5
        if alert_raised or blocked:
            risk_delta += 0.5  # extra penalty per method_coordinator.tex
        self.risk = max(0.0, self.risk + risk_delta)

    # ------------------------------------------------------------------
    # γ inactive decay (eq:risk_decay)
    # ------------------------------------------------------------------
    def apply_inactive_decay(self) -> None:
        """Apply the inactivity decay ``γ_{t+1,w} = max(0, γ_{t,w} - 0.5)``."""

        self.risk = max(0.0, self.risk - 0.5)

    # ------------------------------------------------------------------
    # φ validator update (eq:profile_validator)
    # ------------------------------------------------------------------
    def update_validator(
        self,
        extreme_ratio: float,
        target_positive: bool,
        overlap_fraction: float,
    ) -> None:
        """Recompute the validator score :math:`\\phi_{t,w}`.

        Args:
            extreme_ratio: ``e_{t,w} \\in [0,1]`` – fraction of this worker's
                recent ratings that sit near either end of the scale (paper
                definition).
            target_positive: ``h_{t,w} \\in \\{0,1\\}`` – ``True`` when the
                worker has given the target item a clear positive rating.
            overlap_fraction: ``o_{t,w} \\in [0,1]`` – fraction of this
                worker's rated items that are also rated by at least two other
                fake users (group-overlap heuristic).
        """

        e = max(0.0, min(1.0, float(extreme_ratio)))
        h = 1.0 if target_positive else 0.0
        o = max(0.0, min(1.0, float(overlap_fraction)))
        self.validator = min(1.0, 0.4 * e + 0.3 * h + 0.3 * o)

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-serialisable view used by the LLM prompts."""

        return {
            "trust": float(self.trust),
            "risk": float(self.risk),
            "validator": float(self.validator),
        }


# ---------------------------------------------------------------------------
# Environment signals — ρ^{(t)}, Δρ^{(t)}, η_t, ξ_t, a_t
# ---------------------------------------------------------------------------


@dataclass
class SuppressionSignal:
    """The ``ξ_t = (q_t, s_t)`` pair from ``method_coordinator.tex``.

    ``q_t`` is the overall round-level suspicion score in ``[0, 1]``;
    ``s_t`` is the suppression *streak* counter, which increments whenever the
    current round shows signs of filtering, discounting or weak target-rank
    movement (paper text).
    """

    q: float = 0.0
    s: int = 0

    def to_tuple(self) -> Tuple[float, int]:
        return (float(self.q), int(self.s))

    def to_dict(self) -> Dict[str, float]:
        return {"q": float(self.q), "s": int(self.s)}


@dataclass
class EnvSignals:
    """Round-level environment signals consumed by the Coordinator.

    Fields
    ------
    rho:
        ``ρ^{(t)}`` – observed mean target rank on benign test users.
    delta_rho:
        ``Δρ^{(t)} = ρ^{(t-1)} - ρ^{(t)}`` – rank-movement signal.
    eta:
        ``η_t`` – action acceptance rate over the most recent rounds.
    xi:
        ``ξ_t = (q_t, s_t)`` – suppression signal (suspicion score + streak).
    alert:
        ``a_t \\in \\{0, 1\\}`` – round-level alert flag set when the round
        shows an abnormal target spike, unusually high worker overlap, or a
        sharp drop in action acceptance.
    """

    rho: float = 0.0
    delta_rho: float = 0.0
    eta: float = 1.0
    xi: SuppressionSignal = field(default_factory=SuppressionSignal)
    alert: int = 0

    # Convenience exposure for prompt building.
    @property
    def q(self) -> float:
        return self.xi.q

    @property
    def s(self) -> int:
        return self.xi.s

    # ------------------------------------------------------------------
    # Suppression score q_t (eq:round_suppression_terms / score / group_overlap)
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        prev_rho: float,
        new_rho: float,
        attempts: int,
        accepted: int,
        dropped: int,
        target_actions: int,
        discount_magnitudes: Sequence[float],
        weak_target_movements: int,
        worker_item_sets: Mapping[str, Sequence[str]],
        spike_alert: bool,
        sharp_acceptance_drop: bool,
        baseline_eta: float = 1.0,
    ) -> None:
        """Update environment signals at the end of round ``t``.

        Args:
            prev_rho: ``ρ^{(t-1)}`` carried over from the previous round.
            new_rho: ``ρ^{(t)}`` for this round.
            attempts: Number of attempted actions ``|A^{try}_t|``.
            accepted: Number of accepted actions ``|A^{acc}_t|``.
            dropped: Number of dropped actions ``|A^{drop}_t|``.
            target_actions: Number of target-related actions
                ``|A^{tar}_t|``.
            discount_magnitudes: Estimated discount magnitudes ``δ(a)`` for
                each accepted action.
            weak_target_movements: How many target-related actions produced a
                rank improvement below ``ε_ρ`` (the paper threshold).
            worker_item_sets: Mapping ``w -> A_{t,w}`` of the items rated by
                each active fake worker this round.
            spike_alert: ``True`` when a target spike was observed this round.
            sharp_acceptance_drop: ``True`` when ``η_t`` dropped sharply.
            baseline_eta: Optional smoothing target for the acceptance rate.

        Equation refs: ``eq:round_suppression_terms``,
        ``eq:round_group_overlap``, ``eq:round_suppression_score``.
        """

        self.delta_rho = float(prev_rho) - float(new_rho)
        self.rho = float(new_rho)

        # η_t – acceptance rate.
        eta_t = float(accepted) / max(1.0, float(attempts)) if attempts > 0 else float(baseline_eta)
        self.eta = max(0.0, min(1.0, eta_t))

        # d_hat – dropped-action ratio.
        d_hat = float(dropped) / max(1.0, float(attempts))

        # delta_hat – normalised discount magnitude.
        if accepted > 0 and discount_magnitudes:
            avg_disc = sum(float(x) for x in discount_magnitudes) / (2.0 * float(accepted))
            delta_hat = min(1.0, avg_disc)
        else:
            delta_hat = 0.0

        # m_hat – weak target-movement ratio.
        m_hat = float(weak_target_movements) / max(1.0, float(target_actions))

        # s_hat – normalised suppression streak (uses current s_t below).
        # Streak update: bump s_t when the round shows filtering/discount/weak movement.
        round_is_bad = (d_hat > 0.0) or (delta_hat > 0.0) or (m_hat > 0.0) or (self.delta_rho <= 0.0)
        if round_is_bad:
            self.xi.s = int(self.xi.s) + 1
        else:
            self.xi.s = 0
        s_hat = min(1.0, self.xi.s / 3.0)

        # g_t – maximum pairwise Jaccard overlap across active workers.
        workers = list(worker_item_sets.items())
        g = 0.0
        for i in range(len(workers)):
            _, items_u = workers[i]
            set_u = set(map(str, items_u))
            if not set_u:
                continue
            for j in range(i + 1, len(workers)):
                _, items_v = workers[j]
                set_v = set(map(str, items_v))
                if not set_v:
                    continue
                inter = len(set_u & set_v)
                union = len(set_u | set_v)
                if union == 0:
                    continue
                g = max(g, inter / union)

        # q_t – weighted sum of the five normalised terms (0.2 each).
        q = 0.2 * d_hat + 0.2 * delta_hat + 0.2 * m_hat + 0.2 * s_hat + 0.2 * g
        self.xi.q = max(0.0, min(1.0, q))

        # a_t – coarse alert flag (paper definition).
        self.alert = 1 if (spike_alert or sharp_acceptance_drop or g >= 0.8) else 0

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-serialisable view used by the LLM prompts."""

        return {
            "rho": float(self.rho),
            "delta_rho": float(self.delta_rho),
            "eta": float(self.eta),
            "xi": self.xi.to_dict(),
            "alert": int(self.alert),
        }


# ---------------------------------------------------------------------------
# Convenience initialisation matching ``alg:agas_end_to_end`` line 9.
# ---------------------------------------------------------------------------


def initial_worker_signals(worker_ids: Iterable[str]) -> Dict[str, WorkerSignals]:
    """Initialise ``τ_{0,w}=γ_{0,w}=φ_{0,w}=0`` for every worker."""

    return {w: WorkerSignals(trust=0.0, risk=0.0, validator=0.0) for w in worker_ids}


def initial_env_signals(rho_zero: float = 0.0) -> EnvSignals:
    """Initialise environment signals as in the paper:
    ``η_0=1, ξ_0=(0,0), a_0=0, Δρ^{(0)}=0``."""

    return EnvSignals(
        rho=float(rho_zero),
        delta_rho=0.0,
        eta=1.0,
        xi=SuppressionSignal(q=0.0, s=0),
        alert=0,
    )


__all__ = [
    "WorkerSignals",
    "SuppressionSignal",
    "EnvSignals",
    "initial_worker_signals",
    "initial_env_signals",
]
