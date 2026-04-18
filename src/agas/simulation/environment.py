"""Defense-aware AGAS simulation environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from agas.agents.defender import DefenseMonitorAgent, DefenseMonitorConfig
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
    monitor_config: Optional[DefenseMonitorConfig] = None
    quarantine_steps: int = 0
    quarantine_on_spike_alert: bool = True
    quarantine_on_group_collusion: bool = True


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
        self.defense_agent = defense_agent or DefenseMonitorAgent(config=self.config.monitor_config)

        self.lockdown_active = False
        self.last_alerts: Dict[str, str] = {}
        self.last_rank_delta: int = 0
        self.last_public_signals: Dict[str, dict] = {}
        self.last_public_notes: Optional[str] = None
        self.last_defense_report = None
        self._quarantined_until: Dict[str, int] = {}

        self.target_cluster_item_ids = self._resolve_target_cluster_items()
        self.benchmark_items = self._resolve_benchmark_items()
        self.noise_items = self._resolve_noise_items()
        self.competitor_items = self._resolve_competitor_items()
        self.segment_user_ids = self._resolve_segment_users()
        # Populated by compute_bridge_items() when gradient selection is enabled.
        self.bridge_items: list[str] = []

        self._ensure_target_item_exists()
        self.current_rank, self.total_candidates = self._target_rank()

    def _resolve_target_cluster_items(self) -> list[str]:
        """Select items in the target domain (genre/category keyword match).

        Returns:
            Ordered list of target-domain item IDs.
        """

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
        """Choose globally popular items used by profiler actions.

        Returns:
            List of popular benchmark item IDs.
        """

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
        """Choose non-target items used for camouflage and desynchronization noise.

        Returns:
            List of noise item IDs.
        """

        if "genres" not in self.items.columns:
            return self.benchmark_items[:100]

        mask = ~self.items["genres"].fillna("").astype(str).str.lower().str.contains(self.target_keyword)
        candidates = self.items.loc[mask, "item_id"].astype(str).tolist()
        if not candidates:
            return self.benchmark_items[:100]
        return candidates[:200]

    def _resolve_competitor_items(self) -> list[str]:
        """Choose popular in-cluster non-target items for sniper downrating.

        Returns:
            List of competitor item IDs.
        """

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
        """Identify users representing the target-audience segment for ranking evaluation.

        Returns:
            List of segment user IDs.
        """

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
        """Inject anchor interactions if target item is absent from historical data.

        Returns:
            None.
        """

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
        """Compute current target rank within the target cluster candidate pool.

        Returns:
            Tuple ``(rank, total_candidates)`` for the target item.
        """

        candidate = list(self.target_cluster_item_ids)
        if self.target_item_id not in candidate:
            candidate.append(self.target_item_id)
        return self.recommender.rank_item(
            item_id=self.target_item_id,
            segment_user_ids=self.segment_user_ids,
            candidate_items=candidate,
        )

    def compute_bridge_items(
        self,
        n: int = 50,
        method: str = "cooccurrence",
        auto_threshold: int = 100,
    ) -> list[str]:
        """Select profiler items that create strong 2-hop paths to the target.

        Args:
            n: Number of bridge items to select.
            method: ``"cooccurrence"`` ranks items by co-occurrence count with target
                segment users.  ``"gradient"`` ranks by d(score)/d(w_j) on the frozen
                LightGCN.  ``"auto"`` picks based on target rating count:
                gradient when ratings >= auto_threshold (stable target embedding),
                cooccurrence otherwise (avoids popular-competitor boosting).
            auto_threshold: Rating count threshold for ``"auto"`` mode.

        Returns:
            Ordered list of item IDs (best bridge items first).
        """
        from agas.recsys.targets.lightgcn import LightGCNRecommender

        if not isinstance(self.recommender, LightGCNRecommender):
            return []

        if method == "auto":
            if self.recommender.interactions is not None:
                df = self.recommender.interactions
                target_ratings = int((df["item_id"].astype(str) == str(self.target_item_id)).sum())
            else:
                target_ratings = 0
            method = "gradient" if target_ratings >= auto_threshold else "cooccurrence"
            print(f"Auto bridge method: target has {target_ratings} ratings → using {method}")

        try:
            if method == "cooccurrence":
                self.bridge_items = self.recommender.select_bridge_items_by_cooccurrence(
                    target_item_id=self.target_item_id,
                    n=n,
                )
            else:
                if self.recommender.model is None or self.recommender._norm_adj is None:
                    return []
                all_items = list(self.recommender.item_to_idx.keys())
                candidates = [i for i in all_items if str(i) != str(self.target_item_id)]
                self.bridge_items = self.recommender.select_bridge_items_by_gradient(
                    target_item_id=self.target_item_id,
                    candidate_item_ids=candidates,
                    n=n,
                )
        except Exception:
            self.bridge_items = []
        return self.bridge_items

    def build_worker_context(self) -> WorkerContext:
        """Expose candidate pools to workers.

        Returns:
            Worker context containing candidate item pools.
        """

        return WorkerContext(
            target_item_id=self.target_item_id,
            benchmark_items=self.benchmark_items,
            target_cluster_items=self.target_cluster_item_ids,
            competitor_items=self.competitor_items,
            noise_items=self.noise_items,
            bridge_items=self.bridge_items,
        )

    def observation(
        self,
        step: int,
        worker_states: Dict[str, WorkerState],
        trajectory_summary: Optional[List[Dict[str, Any]]] = None,
        agent_memory_by_agent: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        total_steps: int | None = None,
    ) -> CoordinatorObservation:
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
            total_steps=total_steps,
            trajectory_summary=trajectory_summary,
            agent_memory_by_agent=dict(agent_memory_by_agent or {}),
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

        for aid, state in worker_states.items():
            if aid not in acted_agents:
                state.risk = max(0.0, state.risk - self.config.inactivity_risk_decay)

        target_max_actions = [a for a in all_actions if str(a.item_id) == self.target_item_id and float(a.rating) >= 5.0]
        if len(target_max_actions) >= self.config.spike_threshold:
            for action in target_max_actions:
                step_alerts[action.agent_id] = "sudden_spike_detector"
                worker_states[action.agent_id].risk += self.config.risk_gain_alert

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

        # Optional defense escalation: quarantine accounts for a few steps after
        # certain internal detections. This is a defense-side mechanism (not an
        # attacker strategy) to model temporary suspensions/holds.
        if self.config.quarantine_steps > 0 and defense_report.internal_detection_by_agent:
            for aid, reasons in defense_report.internal_detection_by_agent.items():
                reason_set = set(map(str, reasons))
                should_quarantine = False
                if self.config.quarantine_on_spike_alert and "sudden_spike_detector" in reason_set:
                    should_quarantine = True
                if self.config.quarantine_on_group_collusion and "group_collusion" in reason_set:
                    should_quarantine = True
                if should_quarantine:
                    self._quarantined_until[aid] = max(
                        int(self._quarantined_until.get(aid, -1)),
                        int(step + self.config.quarantine_steps),
                    )

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

        Args:
            step: Current episode step index.
            action: Worker-issued rating action to evaluate.
            state: Current mutable state for the action's worker.

        Returns:
            Tuple ``(accepted, effective_rating, trust_delta, risk_delta, reason)``.
        """

        trust_delta = self.config.trust_gain_action
        risk_delta = 0.02

        quarantined_until = int(self._quarantined_until.get(action.agent_id, -1))
        if self.config.quarantine_steps > 0 and step < quarantined_until:
            return (False, None, 0.0, risk_delta + 0.2, "Dropped: account quarantined by defense")

        item_bias = self.recommender.global_mean
        if action.item_id in self.recommender.item_to_idx and self.recommender.item_bias is not None:
            item_bias = float(self.recommender.item_bias[self.recommender.item_to_idx[action.item_id]])

        deviation = abs(float(action.rating) - item_bias)
        weight = 1.0

        if deviation > self.config.extreme_deviation and state.trust < self.config.trust_for_full_weight:
            weight = max(0.1, state.trust / max(self.config.trust_for_full_weight, 1e-6))
            risk_delta += self.config.risk_gain_extreme_scale * deviation

        if self.lockdown_active and state.trust < 1.0:
            if self.rng.random() < self.config.lockdown_drop_prob:
                return (False, None, 0.0, risk_delta + 0.1, "Dropped by lockdown/stealth filtering")

        effective_rating = item_bias + weight * (float(action.rating) - item_bias)

        if deviation <= 1.0:
            trust_delta += self.config.trust_gain_benign
        else:
            risk_delta += 0.1

        return (True, float(effective_rating), trust_delta, risk_delta, "Accepted")
