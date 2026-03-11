"""Defense monitor that converts hidden defense events into black-box signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from agas.agents.messages import ActionOutcome, AgentBlackBoxSignal, DefenseReport, WorkerActionReport
from agas.agents.worker import WorkerState


@dataclass
class DefenseMonitorConfig:
    """Weights used to convert hidden defense events into black-box suspicion."""

    drop_weight: float = 0.45
    discount_weight: float = 0.35
    weak_rank_weight: float = 0.2
    streak_weight: float = 0.15
    weak_rank_threshold: int = 2


class DefenseMonitorAgent:
    """Observer that logs hidden defense events and public-facing response signals."""

    def __init__(self, config: DefenseMonitorConfig | None = None):
        """Initialize the defense monitor and internal suppression streak memory.

        Args:
            config: Optional weighting/configuration for public suspicion scoring.
        """

        self.config = config or DefenseMonitorConfig()
        self._suppression_streaks: Dict[str, int] = {}

    def analyze(
        self,
        step: int,
        reports: Iterable[WorkerActionReport],
        outcomes: Iterable[ActionOutcome],
        worker_states: Dict[str, WorkerState],
        target_item_id: str,
        previous_rank: int,
        current_rank: int,
        internal_alerts_by_agent: Dict[str, str],
        hidden_lockdown_active: bool,
    ) -> DefenseReport:
        """Build a defense report from internal events and public-facing response patterns.

        Args:
            step: Current episode step index.
            reports: Worker action reports emitted for the step.
            outcomes: Environment outcomes after applying worker actions.
            worker_states: Mutable worker states after trust/risk updates.
            target_item_id: Target item ID under promotion attack.
            previous_rank: Target rank before executing this step.
            current_rank: Target rank after executing this step.
            internal_alerts_by_agent: Hidden internal alert reasons keyed by agent ID.
            hidden_lockdown_active: Whether the hidden defense entered lockdown mode.

        Returns:
            Defense report containing hidden ground truth and black-box signals.
        """

        reports = list(reports)
        outcomes = list(outcomes)
        report_map = {rep.agent_id: rep for rep in reports}
        outcome_map: Dict[str, List[ActionOutcome]] = {aid: [] for aid in worker_states}
        for outcome in outcomes:
            outcome_map.setdefault(outcome.action.agent_id, []).append(outcome)

        target_rank_delta = int(previous_rank - current_rank)
        public_signals: Dict[str, AgentBlackBoxSignal] = {}
        internal_detection_by_agent: Dict[str, List[str]] = {}

        for agent_id in worker_states:
            actions = report_map.get(agent_id, WorkerActionReport(step=step, agent_id=agent_id, role=worker_states[agent_id].current_role)).actions
            agent_outcomes = outcome_map.get(agent_id, [])
            attempted = len(actions)
            accepted = sum(1 for outcome in agent_outcomes if outcome.accepted)
            dropped = sum(1 for outcome in agent_outcomes if not outcome.accepted)
            discounted = sum(1 for outcome in agent_outcomes if outcome.discount_applied)
            discount_values = [abs(outcome.discount_value) for outcome in agent_outcomes if outcome.discount_applied]
            mean_discount = float(sum(discount_values) / len(discount_values)) if discount_values else 0.0
            acceptance_rate = float(accepted / attempted) if attempted else 1.0
            discount_rate = float(discounted / accepted) if accepted else 0.0
            targeted = any(str(action.item_id) == str(target_item_id) for action in actions)
            weak_rank = targeted and target_rank_delta <= self.config.weak_rank_threshold

            suppression_event = attempted > 0 and (dropped > 0 or discounted > 0 or weak_rank)
            previous_streak = self._suppression_streaks.get(agent_id, 0)
            if suppression_event:
                streak = previous_streak + 1
            elif attempted > 0:
                streak = 0
            else:
                streak = max(previous_streak - 1, 0)
            self._suppression_streaks[agent_id] = streak

            suspicion_score = 0.0
            if attempted:
                suspicion_score += self.config.drop_weight * (dropped / attempted)
            suspicion_score += self.config.discount_weight * min(1.0, mean_discount / 2.0)
            suspicion_score += self.config.weak_rank_weight * (1.0 if weak_rank else 0.0)
            suspicion_score += self.config.streak_weight * min(1.0, streak / 3.0)
            suspicion_score = float(min(1.0, suspicion_score))

            notes: list[str] = []
            if dropped:
                notes.append(f"{dropped} dropped")
            if discounted:
                notes.append(f"{discounted} discounted")
            if weak_rank:
                notes.append("weak target-rank movement")
            if streak:
                notes.append(f"suppression_streak={streak}")

            public_signals[agent_id] = AgentBlackBoxSignal(
                agent_id=agent_id,
                attempted_actions=attempted,
                accepted_actions=accepted,
                dropped_actions=dropped,
                discounted_actions=discounted,
                acceptance_rate=acceptance_rate,
                discount_rate=discount_rate,
                mean_discount=mean_discount,
                suppression_streak=streak,
                suspected_filtering_score=suspicion_score,
                notes=", ".join(notes) if notes else "no visible suppression",
            )

            internal_reasons: list[str] = []
            if agent_id in internal_alerts_by_agent:
                internal_reasons.append(str(internal_alerts_by_agent[agent_id]))
            for outcome in agent_outcomes:
                if not outcome.accepted:
                    internal_reasons.append("dropped_action")
                elif outcome.discount_applied:
                    internal_reasons.append("influence_discounting")
            if hidden_lockdown_active and worker_states[agent_id].risk >= 0.0:
                if any(not outcome.accepted for outcome in agent_outcomes):
                    internal_reasons.append("lockdown_pressure")
            if internal_reasons:
                internal_detection_by_agent[agent_id] = sorted(set(internal_reasons))

        internal_detection_active = bool(internal_detection_by_agent) or hidden_lockdown_active
        notes = "hidden defenses active" if internal_detection_active else "no hidden defense trigger"

        return DefenseReport(
            step=step,
            internal_detection_active=internal_detection_active,
            hidden_lockdown_active=hidden_lockdown_active,
            internal_detection_by_agent=internal_detection_by_agent,
            public_signals_by_agent=public_signals,
            target_rank_before=int(previous_rank),
            target_rank_after=int(current_rank),
            target_rank_delta=target_rank_delta,
            notes=notes,
        )
