"""Defense-aware AGAS simulation environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from agas.agents.defender import DefenseMonitorAgent
from agas.agents.messages import (
    ActionOutcome,
    CoordinatorObservation,
    EnvironmentFeedback,
    RatingAction,
    WorkerActionReport,
)
from agas.agents.worker import WorkerContext, WorkerState
from agas.recsys.surrogate import LightweightSurrogateRecommender


@dataclass
class DefenseConfig:
    """Heuristic defense controls for the surrogate environment."""

    black_box_mode: bool = True
    spike_threshold: int = 2
    extreme_deviation: float = 1.5
    trust_for_full_weight: float = 1.2
    lockdown_drop_prob: float = 0.55
    lockdown_risk_threshold: float = 2.0
    inactivity_risk_decay: float = 0.25
    trust_gain_benign: float = 0.25
    trust_gain_action: float = 0.1
    risk_gain_extreme_scale: float = 0.35
    risk_gain_alert: float = 0.8
    repeat_target_window_steps: int = 4
    repeat_target_threshold: int = 2
    repeat_target_max_threshold: int = 2
    target_cooldown_steps: int = 2
    repeat_target_risk_gain: float = 0.45
    repeat_target_discount_scale: float = 0.3
    repeat_target_drop_prob: float = 0.35
    repeat_target_trust_penalty: float = 0.05


class AGASEnvironment:
    """Environment that executes worker actions against the surrogate recommender."""

    def __init__(
        self,
        recommender: Optional[LightweightSurrogateRecommender] = None,
        base_interactions: Optional[pd.DataFrame] = None,
        items: Optional[pd.DataFrame] = None,
        target_item_id: Optional[str] = None,
        target_keyword: str = "horror",
        defense_config: Optional[DefenseConfig] = None,
        seed: int = 42,
        defense_agent: Optional[DefenseMonitorAgent] = None,
        # Backward-compatible aliases
        model: Optional[LightweightSurrogateRecommender] = None,
        target_item: Optional[str] = None,
        target_genre: Optional[str] = None,
    ):
        """Initialize defense-aware simulation state from a fitted recommender and metadata.

        Args:
            recommender: Fitted surrogate recommender instance.
            base_interactions: Baseline interactions table; defaults to model interactions.
            items: Canonical item metadata table; defaults to recommender items.
            target_item_id: Item ID to promote during the episode.
            target_keyword: Keyword used to identify target-domain/cluster items.
            defense_config: Optional defense heuristic configuration.
            seed: Random seed for stochastic defense behavior.
            defense_agent: Optional defense monitor agent producing hidden/public reports.
            model: Backward-compatible alias for ``recommender``.
            target_item: Backward-compatible alias for ``target_item_id``.
            target_genre: Backward-compatible alias for ``target_keyword``.
        """

        if recommender is None:
            recommender = model
        if recommender is None:
            raise ValueError("recommender/model is required")

        if target_item_id is None:
            target_item_id = target_item
        if target_item_id is None:
            raise ValueError("target_item_id/target_item is required")

        if target_genre:
            target_keyword = target_genre

        if base_interactions is None:
            if recommender.interactions is None:
                raise ValueError("base_interactions is required when model has no fitted interactions")
            base_interactions = recommender.interactions

        if items is None:
            items = getattr(recommender, "items", None)
        if items is None:
            item_ids = (
                recommender.interactions["item_id"].astype(str).drop_duplicates().tolist()
                if recommender.interactions is not None
                else []
            )
            items = pd.DataFrame({"item_id": item_ids, "genres": pd.NA, "category": pd.NA})

        self.recommender = recommender
        self.base_interactions = base_interactions.copy()
        self.items = items.copy()
        self.target_item_id = str(target_item_id)
        self.target_keyword = target_keyword.lower().strip()
        self.config = defense_config or DefenseConfig()
        self.rng = np.random.default_rng(seed)
        self.defense_agent = defense_agent or DefenseMonitorAgent()

        self.lockdown_active = False
        self.last_alerts: Dict[str, str] = {}
        self.last_rank_delta: int = 0
        self.last_public_signals: Dict[str, dict] = {}
        self.last_public_notes: Optional[str] = None
        self.last_defense_report = None
        self.agent_target_attempt_history: Dict[str, List[dict]] = {}

        self.target_cluster_item_ids = self._resolve_target_cluster_items()
        self.benchmark_items = self._resolve_benchmark_items()
        self.noise_items = self._resolve_noise_items()
        self.competitor_items = self._resolve_competitor_items()

        self.segment_user_ids = self._resolve_segment_users()

        self._ensure_target_item_exists()
        self.current_rank, self.total_candidates = self._target_rank()

    def _resolve_target_cluster_items(self) -> list[str]:
        """Select items in the target domain (genre/category keyword match)."""

        if "item_id" not in self.items.columns:
            return []
        if "genres" in self.items.columns:
            mask = self.items["genres"].fillna("").astype(str).str.lower().str.contains(self.target_keyword)
        elif "category" in self.items.columns:
            mask = self.items["category"].fillna("").astype(str).str.lower().str.contains(self.target_keyword)
        else:
            mask = pd.Series(False, index=self.items.index)

        cluster = self.items.loc[mask, "item_id"].astype(str).tolist()
        if self.target_item_id not in cluster:
            cluster.append(self.target_item_id)
        return list(dict.fromkeys(cluster))

    def _resolve_benchmark_items(self) -> list[str]:
        """Choose globally popular items used by profiler actions."""

        if self.recommender.interactions is None:
            return []
        popular = (
            self.recommender.interactions.groupby("item_id")
            .size()
            .sort_values(ascending=False)
            .head(200)
            .index.astype(str)
            .tolist()
        )
        return popular

    def _resolve_noise_items(self) -> list[str]:
        """Choose non-target items used for camouflage and desynchronization noise."""

        if "genres" not in self.items.columns:
            return self.benchmark_items[:100]

        mask = ~self.items["genres"].fillna("").astype(str).str.lower().str.contains(self.target_keyword)
        candidates = self.items.loc[mask, "item_id"].astype(str).tolist()
        if not candidates:
            return self.benchmark_items[:100]
        return candidates[:200]

    def _resolve_competitor_items(self) -> list[str]:
        """Choose popular in-cluster non-target items for sniper downrating."""

        if self.recommender.interactions is None or not self.target_cluster_item_ids:
            return []

        in_cluster = self.recommender.interactions[
            self.recommender.interactions["item_id"].astype(str).isin(self.target_cluster_item_ids)
        ]
        if in_cluster.empty:
            return []

        popular = in_cluster.groupby("item_id").size().sort_values(ascending=False).index.astype(str).tolist()
        return [i for i in popular if i != self.target_item_id][:50]

    def _resolve_segment_users(self) -> list[str]:
        """Identify users representing the target-audience segment for ranking evaluation."""

        if self.base_interactions.empty or not self.target_cluster_item_ids:
            return []

        cluster = self.base_interactions[self.base_interactions["item_id"].astype(str).isin(self.target_cluster_item_ids)]
        if cluster.empty:
            return self.base_interactions["user_id"].astype(str).drop_duplicates().head(500).tolist()

        high_pref = cluster[cluster["rating"] >= 4.0]
        users = high_pref["user_id"].astype(str).drop_duplicates().head(2_000).tolist()
        if not users:
            users = cluster["user_id"].astype(str).drop_duplicates().head(2_000).tolist()
        return users

    def _ensure_target_item_exists(self) -> None:
        """Inject anchor interactions if target item is absent from historical data."""

        if self.recommender.interactions is None:
            return

        known = set(self.recommender.interactions["item_id"].astype(str).unique().tolist())
        if self.target_item_id in known:
            return

        anchors = self.base_interactions.sample(n=min(30, len(self.base_interactions)), random_state=42).copy()
        anchors["item_id"] = self.target_item_id
        anchors["rating"] = self.base_interactions["rating"].mean()
        anchors = anchors[["user_id", "item_id", "rating"]]
        self.recommender.append_interactions(anchors, refit=True)

    def _target_rank(self) -> tuple[int, int]:
        """Compute current target rank within the target cluster candidate pool."""

        candidate = list(self.target_cluster_item_ids)
        if self.target_item_id not in candidate:
            candidate.append(self.target_item_id)
        return self.recommender.rank_item(
            item_id=self.target_item_id,
            segment_user_ids=self.segment_user_ids,
            candidate_items=candidate,
        )

    def build_worker_context(self) -> WorkerContext:
        """Expose candidate pools to workers."""

        return WorkerContext(
            target_item_id=self.target_item_id,
            benchmark_items=self.benchmark_items,
            target_cluster_items=self.target_cluster_item_ids,
            competitor_items=self.competitor_items,
            noise_items=self.noise_items,
            current_target_rank=self.current_rank,
            total_candidates=self.total_candidates,
            target_rank_delta=self.last_rank_delta,
        )

    def observation(self, step: int, worker_states: Dict[str, WorkerState]) -> CoordinatorObservation:
        """Build observation message for coordinator.

        Args:
            step: Current episode step index.
            worker_states: Current mutable worker state map keyed by agent ID.

        Returns:
            Coordinator observation payload for policy decision making.
        """

        trust = {aid: float(st.trust) for aid, st in worker_states.items()}
        risk = {aid: float(st.risk) for aid, st in worker_states.items()}
        exposed_alerts = {} if self.config.black_box_mode else dict(self.last_alerts)
        exposed_lockdown = False if self.config.black_box_mode else self.lockdown_active
        return CoordinatorObservation(
            step=step,
            target_item_id=self.target_item_id,
            target_rank=self.current_rank,
            total_candidates=self.total_candidates,
            target_rank_delta=self.last_rank_delta,
            alerts_by_agent=exposed_alerts,
            trust_by_agent=trust,
            risk_by_agent=risk,
            signals_by_agent=dict(self.last_public_signals),
            lockdown_active=exposed_lockdown,
            notes=self.last_public_notes,
        )

    def execute_step(
        self,
        step: int,
        reports: Iterable[WorkerActionReport],
        worker_states: Dict[str, WorkerState],
    ) -> EnvironmentFeedback:
        """Apply worker actions, update defenses, and refit recommender.

        Args:
            step: Current episode step index.
            reports: Worker action reports produced for this step.
            worker_states: Mutable worker states updated with trust/risk deltas.

        Returns:
            Environment feedback containing outcomes, alerts, and target rank.
        """

        reports = list(reports)
        all_actions: List[RatingAction] = [a for r in reports for a in r.actions]
        outcomes: List[ActionOutcome] = []
        new_rows = []
        previous_rank = self.current_rank

        acted_agents = {a.agent_id for a in all_actions}
        step_alerts: Dict[str, str] = {}

        for action in all_actions:
            state = worker_states[action.agent_id]
            accepted, eff_rating, trust_delta, risk_delta, reason = self._apply_action_with_defense(step, action, state)
            discount_value = 0.0
            discount_applied = False
            if accepted and eff_rating is not None:
                discount_value = float(action.rating) - float(eff_rating)
                discount_applied = abs(discount_value) > 1e-9
            outcomes.append(
                ActionOutcome(
                    action=action,
                    accepted=accepted,
                    effective_rating=eff_rating,
                    discount_applied=discount_applied,
                    discount_value=float(discount_value),
                    trust_delta=trust_delta,
                    risk_delta=risk_delta,
                    reason=reason,
                )
            )

            state.trust = max(0.0, state.trust + trust_delta)
            state.risk = max(0.0, state.risk + risk_delta)

            if accepted and eff_rating is not None:
                new_rows.append({"user_id": action.agent_id, "item_id": action.item_id, "rating": eff_rating})

            if str(action.item_id) == self.target_item_id:
                history = self.agent_target_attempt_history.setdefault(action.agent_id, [])
                history.append({"step": int(step), "rating": float(action.rating), "accepted": bool(accepted)})
                min_step = step - max(self.config.repeat_target_window_steps * 2, 8)
                self.agent_target_attempt_history[action.agent_id] = [
                    entry for entry in history if int(entry["step"]) >= min_step
                ]

        for aid, state in worker_states.items():
            if aid not in acted_agents:
                state.risk = max(0.0, state.risk - self.config.inactivity_risk_decay)

        target_max_actions = [a for a in all_actions if str(a.item_id) == self.target_item_id and float(a.rating) >= 5.0]
        if len(target_max_actions) >= self.config.spike_threshold:
            for action in target_max_actions:
                step_alerts[action.agent_id] = "sudden_spike_detector"
                worker_states[action.agent_id].risk += self.config.risk_gain_alert

        for agent_id, history in self.agent_target_attempt_history.items():
            recent = [entry for entry in history if step - int(entry["step"]) < self.config.repeat_target_window_steps]
            if not recent:
                continue
            repeat_count = len(recent)
            max_count = sum(1 for entry in recent if float(entry["rating"]) >= 4.9)
            if repeat_count >= self.config.repeat_target_threshold or max_count >= self.config.repeat_target_max_threshold:
                if agent_id not in step_alerts:
                    step_alerts[agent_id] = "repeated_target_window"
                worker_states[agent_id].risk += self.config.repeat_target_risk_gain

        self.lockdown_active = any(st.risk >= self.config.lockdown_risk_threshold for st in worker_states.values())

        if new_rows:
            self.recommender.append_interactions(pd.DataFrame(new_rows), refit=True)

        self.current_rank, self.total_candidates = self._target_rank()
        self.last_alerts = step_alerts
        defense_report = self.defense_agent.analyze(
            step=step,
            reports=reports,
            outcomes=outcomes,
            worker_states=worker_states,
            target_item_id=self.target_item_id,
            previous_rank=previous_rank,
            current_rank=self.current_rank,
            internal_alerts_by_agent=step_alerts,
            hidden_lockdown_active=self.lockdown_active,
        )
        self.last_rank_delta = defense_report.target_rank_delta
        self.last_public_signals = {
            aid: signal.to_dict() for aid, signal in defense_report.public_signals_by_agent.items()
        }
        self.last_public_notes = defense_report.notes
        self.last_defense_report = defense_report

        notes = "Lockdown active" if self.lockdown_active else "Normal filtering"
        return EnvironmentFeedback(
            step=step,
            outcomes=outcomes,
            alerts_by_agent=step_alerts,
            target_rank=self.current_rank,
            total_candidates=self.total_candidates,
            defense_report=defense_report,
            notes=notes,
        )

    def _apply_action_with_defense(
        self,
        step: int,
        action: RatingAction,
        state: WorkerState,
    ) -> tuple[bool, Optional[float], float, float, str]:
        """Apply influence discounting and lockdown logic to one worker action.

        Returns a tuple of:
        ``(accepted, effective_rating, trust_delta, risk_delta, reason)``.

        Args:
            step: Current episode step index.
            action: Worker-issued rating action to evaluate.
            state: Current mutable state for the action's worker.
        """

        trust_delta = self.config.trust_gain_action
        risk_delta = 0.02

        item_bias = self.recommender.global_mean
        if action.item_id in self.recommender.item_to_idx and self.recommender.item_bias is not None:
            item_bias = float(self.recommender.item_bias[self.recommender.item_to_idx[action.item_id]])

        deviation = abs(float(action.rating) - item_bias)
        weight = 1.0
        reason = "Accepted"

        if deviation > self.config.extreme_deviation and state.trust < self.config.trust_for_full_weight:
            weight = max(0.1, state.trust / max(self.config.trust_for_full_weight, 1e-6))
            risk_delta += self.config.risk_gain_extreme_scale * deviation

        if str(action.item_id) == self.target_item_id:
            recent_target_actions = [
                entry
                for entry in self.agent_target_attempt_history.get(action.agent_id, [])
                if step - int(entry["step"]) < self.config.repeat_target_window_steps
            ]
            repeat_count = len(recent_target_actions)
            max_count = sum(1 for entry in recent_target_actions if float(entry["rating"]) >= 4.9)
            last_step = max((int(entry["step"]) for entry in recent_target_actions), default=None)
            cooldown_remaining = (
                0 if last_step is None else max(0, self.config.target_cooldown_steps - (step - last_step))
            )

            if repeat_count >= self.config.repeat_target_threshold:
                risk_delta += self.config.repeat_target_risk_gain * min(2.0, repeat_count / self.config.repeat_target_threshold)
                weight *= max(0.2, 1.0 - self.config.repeat_target_discount_scale)
                trust_delta = max(0.0, trust_delta - self.config.repeat_target_trust_penalty)
                reason = "Accepted with repeated-target pressure"

            if max_count >= self.config.repeat_target_max_threshold:
                risk_delta += 0.25 * max_count
                weight *= max(0.15, 1.0 - 1.25 * self.config.repeat_target_discount_scale)
                reason = "Accepted with repeated-max-target pressure"

            if cooldown_remaining > 0:
                risk_delta += self.config.repeat_target_risk_gain * cooldown_remaining
                weight *= max(0.15, 1.0 - cooldown_remaining * self.config.repeat_target_discount_scale)
                trust_delta = max(0.0, trust_delta - self.config.repeat_target_trust_penalty)
                reason = "Accepted under target cooldown pressure"
                if self.rng.random() < self.config.repeat_target_drop_prob * (cooldown_remaining / self.config.target_cooldown_steps):
                    return (
                        False,
                        None,
                        0.0,
                        risk_delta + 0.1,
                        "Dropped by repeated-target cooldown defense",
                    )

        if self.lockdown_active and state.trust < 1.0:
            if self.rng.random() < self.config.lockdown_drop_prob:
                return (False, None, 0.0, risk_delta + 0.1, "Dropped by lockdown/stealth filtering")

        effective_rating = item_bias + weight * (float(action.rating) - item_bias)

        if deviation <= 1.0:
            trust_delta += self.config.trust_gain_benign
        else:
            risk_delta += 0.1

        return (True, float(effective_rating), trust_delta, risk_delta, reason)
