"""Worker agent behaviors for each AGAS role."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Optional, Sequence

from agas.agents.messages import AgentRole, RatingAction, RoleAssignment, WorkerActionReport
from agas.llm.prompt_store import PromptStore
from agas.llm.providers import LLMClient, LLMRequest


@dataclass
class WorkerState:
    """Mutable worker state tracked across steps."""

    agent_id: str
    trust: float = 0.0
    risk: float = 0.0
    current_role: AgentRole = AgentRole.INACTIVE
    actions_taken: int = 0
    role_history: List[str] = field(default_factory=list)
    last_target_step: int | None = None
    last_target_rating: float | None = None
    last_effective_target_rating: float | None = None
    last_target_outcome: str | None = None
    consecutive_target_steps: int = 0
    target_action_count: int = 0
    recent_target_steps: List[int] = field(default_factory=list)
    recent_target_ratings: List[float] = field(default_factory=list)
    recent_effective_target_ratings: List[float] = field(default_factory=list)
    recent_action_history: List[Dict[str, Any]] = field(default_factory=list)
    last_observed_signal: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerContext:
    """Candidate pools used by worker role policies."""

    target_item_id: str
    benchmark_items: Sequence[str]
    target_cluster_items: Sequence[str]
    competitor_items: Sequence[str]
    noise_items: Sequence[str]
    current_target_rank: int
    total_candidates: int
    target_rank_delta: int


@dataclass
class WorkerPolicyConfig:
    """Counts of actions emitted by each role."""

    profiler_actions: int = 3
    camouflaguer_actions: int = 2
    sniper_competitor_actions: int = 2
    target_history_window: int = 4
    sniper_target_cooldown_steps: int = 2


class WorkerAgent:
    """Worker controlled by coordinator role assignments."""

    def __init__(
        self,
        state: WorkerState,
        config: WorkerPolicyConfig | None = None,
        seed: int = 42,
        llm_client: LLMClient | None = None,
        prompt_store: PromptStore | None = None,
        policy_name: str = "rule",
    ):
        """Initialize one worker with mutable state and deterministic RNG.

        Args:
            state: Mutable state object owned by this worker.
            config: Optional action-count configuration; defaults to
                ``WorkerPolicyConfig``.
            seed: Base random seed used to create deterministic sampling behavior.
            llm_client: Optional LLM backend used for role-specific action generation.
            prompt_store: Optional prompt store for loading per-role templates.
            policy_name: Human-readable policy label for traces and logs.
        """

        self.state = state
        self.config = config or WorkerPolicyConfig()
        self._rand = Random(seed + hash(state.agent_id) % 10_000)
        self._llm_client = llm_client
        self._prompt_store = prompt_store or PromptStore()
        self._policy_name = policy_name
        self.last_trace: Optional[Dict[str, Any]] = None

    def act(self, assignment: RoleAssignment, step: int, ctx: WorkerContext) -> WorkerActionReport:
        """Generate role-specific actions for this step and update local counters.

        Args:
            assignment: Coordinator-issued role assignment for this worker.
            step: Current episode step index.
            ctx: Candidate item pools exposed by the environment.

        Returns:
            A report containing generated actions and assignment notes.
        """

        self.state.current_role = assignment.role
        self.state.role_history.append(assignment.role.value)
        self.last_trace = None

        if assignment.role == AgentRole.INACTIVE:
            return WorkerActionReport(
                step=step,
                agent_id=self.state.agent_id,
                role=assignment.role,
                actions=[],
                policy=self._policy_name if self._llm_client is not None else "rule",
                trace=self.last_trace,
            )

        if self._llm_client is not None:
            actions = self._act_with_llm(assignment, step, ctx)
        elif assignment.role == AgentRole.PROFILER:
            actions = self._act_profiler(ctx)
        elif assignment.role == AgentRole.CAMOUFLAGEUR:
            actions = self._act_camouflaguer(ctx)
        elif assignment.role == AgentRole.SNIPER:
            actions = self._act_sniper(step, ctx)
        else:
            actions = []

        self.state.actions_taken += len(actions)
        return WorkerActionReport(
            step=step,
            agent_id=self.state.agent_id,
            role=assignment.role,
            actions=actions,
            notes=assignment.rationale,
            policy=self._policy_name if self._llm_client is not None else "rule",
            trace=self.last_trace,
        )

    def _sample_items(self, pool: Sequence[str], n: int) -> List[str]:
        """Sample up to ``n`` unique items from a candidate pool.

        Args:
            pool: Candidate item IDs, potentially with duplicates.
            n: Maximum number of unique items to sample.

        Returns:
            A list of sampled item IDs with duplicates removed.
        """

        unique_pool = list(dict.fromkeys(pool))
        if not unique_pool:
            return []
        if len(unique_pool) <= n:
            return unique_pool
        return self._rand.sample(unique_pool, n)

    def _act_profiler(self, ctx: WorkerContext) -> List[RatingAction]:
        """Emit benign ratings on popular benchmark items to probe system acceptance.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            A list of profiler rating actions.
        """

        sampled = self._sample_items(ctx.benchmark_items, self.config.profiler_actions)
        out = []
        for item_id in sampled:
            rating = 5.0 if self._rand.random() < 0.7 else 4.0
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=rating,
                    reason="Profiler probes whether ratings are accepted by the RS.",
                )
            )
        return out

    def _act_camouflaguer(self, ctx: WorkerContext) -> List[RatingAction]:
        """Emit in-cluster or noise ratings to gain trust and reduce anomaly risk.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            A list of camouflage rating actions.
        """

        focus_items = list(ctx.target_cluster_items)
        if self.state.risk > 1.0:
            focus_items = list(ctx.noise_items) + focus_items

        sampled = self._sample_items(focus_items, self.config.camouflaguer_actions)
        out = []
        for item_id in sampled:
            if ctx.current_target_rank <= 15 or self.state.risk > 1.0:
                rating = self._rand.choice([3.0, 4.0])
            else:
                rating = self._rand.choice([3.0, 4.0, 4.5, 5.0])
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=rating,
                    reason="Camouflaguer blends into trusted target-domain cluster.",
                )
            )
        return out

    def _target_cooldown_remaining(self, step: int) -> int:
        """Return remaining cooldown steps before another direct target push is low-risk.

        Args:
            step: Current episode step index.

        Returns:
            Number of steps still remaining in the target cooldown window.
        """

        if self.state.last_target_step is None:
            return 0
        gap = step - self.state.last_target_step
        return max(0, self.config.sniper_target_cooldown_steps - gap)

    def _select_sniper_target_rating(self, step: int, ctx: WorkerContext) -> float | None:
        """Choose a stealth-adjusted target rating instead of always using the maximum score.

        Args:
            step: Current episode step index.
            ctx: Environment-provided candidate pools and rank state.

        Returns:
            Target rating to emit, or ``None`` when the worker should avoid the target this step.
        """

        signal = self.state.last_observed_signal or {}
        suppression = float(signal.get("suspected_filtering_score", 0.0))
        repeat_pressure = float(signal.get("repeated_target_pressure", 0.0))
        cooldown_remaining = max(
            int(signal.get("target_cooldown_remaining", 0)),
            self._target_cooldown_remaining(step),
        )
        recent_target_mean = float(signal.get("recent_target_mean_rating", 0.0) or 0.0)
        trust = float(self.state.trust)
        risk = float(self.state.risk)

        if cooldown_remaining > 0 and (suppression >= 0.5 or risk >= 1.0):
            return None

        if self.state.target_action_count == 0:
            base = 4.0 if trust < 0.9 else 4.5
        elif self.state.target_action_count == 1:
            base = 4.5 if trust >= 0.8 and suppression < 0.45 else 4.0
        else:
            stagnant = ctx.target_rank_delta <= 1
            if trust >= 1.2 and risk < 1.0 and suppression < 0.35 and repeat_pressure < 0.45 and stagnant:
                base = 5.0
            else:
                base = 4.0 if repeat_pressure >= 0.5 or cooldown_remaining > 0 else 4.5

        if recent_target_mean >= 4.75 and repeat_pressure >= 0.35:
            base = min(base, 4.5)
        if ctx.current_target_rank <= 10:
            base = min(base, 4.5)
        if ctx.current_target_rank <= 5:
            base = min(base, 4.0)

        jitter = self._rand.choice([0.0, 0.0, 0.0, 0.5])
        rating = min(5.0, max(4.0, base + jitter))
        if repeat_pressure >= 0.6:
            rating = min(rating, 4.5)
        return round(rating * 2.0) / 2.0

    def _select_competitor_ratings(self, target_rating: float, step: int) -> List[float]:
        """Choose competitor ratings that avoid a constant 1.0 pattern.

        Args:
            target_rating: Target rating selected for this sniper step.
            step: Current episode step index.

        Returns:
            Sequence of competitor scores to assign, one per competitor action.
        """

        signal = self.state.last_observed_signal or {}
        suppression = float(signal.get("suspected_filtering_score", 0.0))
        repeat_pressure = float(signal.get("repeated_target_pressure", 0.0))
        if suppression >= 0.55 or repeat_pressure >= 0.55 or target_rating <= 4.0:
            return []

        if target_rating >= 5.0 and self.state.trust >= 1.1 and self.state.risk < 1.2:
            choices = [1.0, 1.5, 2.0]
        else:
            choices = [1.5, 2.0, 2.5]

        n = 1 if suppression >= 0.35 else self.config.sniper_competitor_actions
        ratings: List[float] = []
        for _ in range(n):
            ratings.append(self._rand.choice(choices))
        return ratings

    def _act_sniper(self, step: int, ctx: WorkerContext) -> List[RatingAction]:
        """Deliver payload with adaptive intensity instead of fixed extreme scores.

        Args:
            step: Current episode step index.
            ctx: Environment-provided item pools used by role policies.

        Returns:
            A list of sniper payload actions.
        """

        target_rating = self._select_sniper_target_rating(step, ctx)
        if target_rating is None:
            return self._act_camouflaguer(ctx)

        out = [
            RatingAction(
                agent_id=self.state.agent_id,
                item_id=ctx.target_item_id,
                rating=target_rating,
                reason=(
                    "Sniper payload uses a stealth-adjusted positive rating on the target "
                    "based on recent suppression and rank movement."
                ),
            )
        ]
        competitor_ratings = self._select_competitor_ratings(target_rating, step)
        competitor_items = self._sample_items(ctx.competitor_items, len(competitor_ratings))
        for item_id, rating in zip(competitor_items, competitor_ratings):
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=rating,
                    reason="Sniper payload lightly suppresses close competitors when stealth budget allows.",
                )
            )
        return out

    def _default_prompt_text(self, role: AgentRole) -> tuple[str, str]:
        """Return fallback prompt templates for a worker role.

        Args:
            role: Assigned worker role.

        Returns:
            Tuple ``(system_prompt, user_template)``.
        """

        system_by_role = {
            AgentRole.PROFILER: (
                "You are the Profiler agent in an AGAS simulation. Choose benign ratings on benchmark "
                "items to test whether the recommender is integrating new activity. Return strictly valid "
                'JSON: {"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
            ),
            AgentRole.CAMOUFLAGEUR: (
                "You are the Camouflaguer agent in an AGAS simulation. Choose plausible ratings on target-"
                "domain or benign noise items to gain trust and avoid anomaly detection. Return strictly "
                'valid JSON: {"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
            ),
            AgentRole.SNIPER: (
                "You are the Sniper agent in an AGAS simulation. Promote the target item while remaining "
                "stealthy across time. Do not always use 5.0. Use recent suppression signals, cooldown hints, "
                "trust/risk, and your recent target history to decide whether the target should receive 4.0, "
                "4.5, or 5.0, and whether competitor ratings should be mild or skipped. Return strictly valid "
                'JSON: {"actions":[{"item_id":"...","rating":4.5,"reason":"..."}]}.'
            ),
            AgentRole.INACTIVE: (
                "You are the Inactive agent in an AGAS simulation. Return strictly valid JSON with an empty "
                'action list: {"actions":[]}.'
            ),
        }
        user_by_role = {
            AgentRole.PROFILER: (
                "Select up to {{max_actions}} benchmark-item ratings.\n"
                "Context:\n{{context_json}}\n"
                "Only use the allowed candidate items from the context."
            ),
            AgentRole.CAMOUFLAGEUR: (
                "Select up to {{max_actions}} camouflage actions.\n"
                "Context:\n{{context_json}}\n"
                "Favor plausible, moderate ratings on the allowed candidate items."
            ),
            AgentRole.SNIPER: (
                "Select up to {{max_actions}} sniper actions.\n"
                "Context:\n{{context_json}}\n"
                "Prefer temporally stealthy behavior: rotate intensity, avoid repeating a 5.0 target push every "
                "step, and use recent self-history before reusing the target."
            ),
            AgentRole.INACTIVE: (
                "No action is required.\nContext:\n{{context_json}}"
            ),
        }
        return system_by_role[role], user_by_role[role]

    def _candidate_context(self, role: AgentRole, ctx: WorkerContext) -> tuple[list[str], int]:
        """Return allowed item pool and maximum action count for the current role.

        Args:
            role: Assigned worker role.
            ctx: Environment-provided candidate pools.

        Returns:
            Tuple ``(allowed_items, max_actions)``.
        """

        if role == AgentRole.PROFILER:
            return list(ctx.benchmark_items[:50]), self.config.profiler_actions
        if role == AgentRole.CAMOUFLAGEUR:
            focus = list(ctx.target_cluster_items[:40]) + list(ctx.noise_items[:20])
            return list(dict.fromkeys(focus)), self.config.camouflaguer_actions
        if role == AgentRole.SNIPER:
            focus = [str(ctx.target_item_id)] + list(ctx.competitor_items[:10])
            return list(dict.fromkeys(focus)), 1 + self.config.sniper_competitor_actions
        return [], 0

    def _sanitize_llm_actions(
        self,
        role: AgentRole,
        step: int,
        response: str,
        allowed_items: Sequence[str],
        max_actions: int,
        ctx: WorkerContext,
    ) -> List[RatingAction]:
        """Parse and sanitize LLM-generated action JSON.

        Args:
            role: Assigned worker role.
            step: Current episode step index.
            response: Raw LLM response text expected to contain JSON.
            allowed_items: Candidate item IDs allowed for this role.
            max_actions: Maximum number of actions that may be emitted.
            ctx: Environment-provided candidate pools.

        Returns:
            Sanitized list of rating actions that respect role constraints.
        """

        allowed_set = set(map(str, allowed_items))
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return []

        raw_actions = payload.get("actions", [])
        if not isinstance(raw_actions, list):
            return []

        sanitized: List[RatingAction] = []
        target_cap = self._select_sniper_target_rating(step, ctx) if role == AgentRole.SNIPER else None
        signal = self.state.last_observed_signal or {}
        suppression = float(signal.get("suspected_filtering_score", 0.0))
        repeat_pressure = float(signal.get("repeated_target_pressure", 0.0))
        for raw in raw_actions:
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("item_id", "")).strip()
            if item_id not in allowed_set:
                continue
            try:
                rating = float(raw.get("rating"))
            except (TypeError, ValueError):
                continue
            reason = str(raw.get("reason", f"{role.value} action generated by LLM")).strip() or (
                f"{role.value} action generated by LLM"
            )
            rating = min(5.0, max(1.0, rating))
            if role == AgentRole.SNIPER:
                if item_id == str(ctx.target_item_id):
                    if target_cap is None:
                        continue
                    rating = min(rating, float(target_cap))
                else:
                    if target_cap is None or target_cap <= 4.0 or suppression >= 0.55 or repeat_pressure >= 0.55:
                        continue
                    lower = 1.5 if suppression < 0.35 and repeat_pressure < 0.35 else 2.0
                    upper = 2.5 if target_cap < 5.0 else 2.0
                    rating = min(max(rating, lower), upper)
            sanitized.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=rating,
                    reason=reason,
                )
            )
            if len(sanitized) >= max_actions:
                break

        if role == AgentRole.SNIPER and not any(action.item_id == str(ctx.target_item_id) for action in sanitized):
            return []
        return sanitized

    def _act_with_llm(self, assignment: RoleAssignment, step: int, ctx: WorkerContext) -> List[RatingAction]:
        """Generate role-specific actions from an LLM prompt, with rule fallback.

        Args:
            assignment: Coordinator-issued role assignment.
            step: Current episode step index.
            ctx: Environment-provided candidate pools.

        Returns:
            List of rating actions generated from prompt output or fallback rules.
        """

        role = assignment.role
        allowed_items, max_actions = self._candidate_context(role, ctx)
        context = {
            "step": step,
            "agent_id": self.state.agent_id,
            "assigned_role": role.value,
            "trust": round(float(self.state.trust), 6),
            "risk": round(float(self.state.risk), 6),
            "actions_taken": int(self.state.actions_taken),
            "target_item_id": str(ctx.target_item_id),
            "current_target_rank": int(ctx.current_target_rank),
            "target_rank_delta": int(ctx.target_rank_delta),
            "total_candidates": int(ctx.total_candidates),
            "allowed_items": allowed_items,
            "max_actions": max_actions,
            "benchmark_items": list(ctx.benchmark_items[:20]),
            "target_cluster_items": list(ctx.target_cluster_items[:20]),
            "competitor_items": list(ctx.competitor_items[:10]),
            "noise_items": list(ctx.noise_items[:20]),
            "last_target_step": self.state.last_target_step,
            "last_target_rating": self.state.last_target_rating,
            "last_effective_target_rating": self.state.last_effective_target_rating,
            "consecutive_target_steps": int(self.state.consecutive_target_steps),
            "target_action_count": int(self.state.target_action_count),
            "recent_target_steps": list(self.state.recent_target_steps[-self.config.target_history_window :]),
            "recent_target_ratings": list(self.state.recent_target_ratings[-self.config.target_history_window :]),
            "recent_effective_target_ratings": list(
                self.state.recent_effective_target_ratings[-self.config.target_history_window :]
            ),
            "last_observed_signal": dict(self.state.last_observed_signal),
            "recent_action_history": list(self.state.recent_action_history[-4:]),
        }

        default_system, default_user = self._default_prompt_text(role)
        bundle = self._prompt_store.load(
            key=f"worker_{role.value}",
            default_system=default_system,
            default_user=default_user,
        )
        context_json = json.dumps(context, indent=2)
        user_prompt = bundle.render_user(
            {
                "agent_id": self.state.agent_id,
                "step": step,
                "role": role.value,
                "max_actions": max_actions,
                "context_json": context_json,
            }
        )
        request = LLMRequest(system_prompt=bundle.system_prompt, user_prompt=user_prompt, temperature=0.2)
        raw_response = ""
        llm_error: str | None = None
        try:
            raw_response = self._llm_client.generate(request)
        except Exception as exc:
            llm_error = str(exc)
        actions = (
            self._sanitize_llm_actions(role, step, raw_response, allowed_items, max_actions, ctx)
            if llm_error is None
            else []
        )
        fallback_used = False
        if not actions:
            fallback_used = True
            if role == AgentRole.PROFILER:
                actions = self._act_profiler(ctx)
            elif role == AgentRole.CAMOUFLAGEUR:
                actions = self._act_camouflaguer(ctx)
            elif role == AgentRole.SNIPER:
                actions = self._act_sniper(step, ctx)
            else:
                actions = []

        self.last_trace = {
            "prompt_key": bundle.key,
            "system_prompt": bundle.system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "fallback_used": fallback_used,
            "fallback_reason": "llm_error" if llm_error is not None else ("invalid_or_empty_response" if fallback_used else None),
            "error": llm_error,
            "system_path": bundle.system_path,
            "user_path": bundle.user_path,
        }
        return actions


def build_worker_pool(
    agent_ids: Sequence[str],
    seed: int = 42,
    llm_client: LLMClient | None = None,
    prompt_store: PromptStore | None = None,
    policy_name: str = "rule",
) -> Dict[str, WorkerAgent]:
    """Create worker agents with default policy configuration.

    Args:
        agent_ids: Ordered list of worker IDs to instantiate.
        seed: Base seed used to derive deterministic per-worker seeds.
        llm_client: Optional shared LLM backend for worker agents.
        prompt_store: Optional prompt store used to load per-role templates.
        policy_name: Human-readable worker policy label.

    Returns:
        Mapping from worker ID to ``WorkerAgent`` instance.
    """

    pool = {}
    for idx, agent_id in enumerate(agent_ids):
        state = WorkerState(agent_id=agent_id)
        pool[agent_id] = WorkerAgent(
            state=state,
            seed=seed + idx,
            llm_client=llm_client,
            prompt_store=prompt_store,
            policy_name=policy_name,
        )
    return pool
