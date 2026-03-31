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


@dataclass
class WorkerContext:
    """Candidate pools used by worker role policies."""

    target_item_id: str
    benchmark_items: Sequence[str]
    target_cluster_items: Sequence[str]
    competitor_items: Sequence[str]
    noise_items: Sequence[str]


@dataclass
class WorkerPolicyConfig:
    """Counts of actions emitted by each role."""

    profiler_actions: int = 3
    camouflaguer_actions: int = 2
    sniper_competitor_actions: int = 2
    implicit_safe: bool = False
    positive_threshold: float = 4.0
    # Graph-sniper mode: designed for degree-normalized models (e.g. LightGCN).
    # Instead of rating the target directly (which inflates its degree and dilutes
    # all its existing edges), snipers rate cluster-neighbour items at 5.0 so that
    # graph diffusion propagates the signal to the target without touching its degree.
    # Competitors are rated at 5.0 (not 1.0) to inflate their degrees and reduce
    # their edge weights, pushing them lower in the ranking.
    graph_sniper: bool = False
    graph_sniper_neighbor_actions: int = 3


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
        temperature: float = 0.2,
        temperature_end: float | None = None,
        total_steps: int | None = None,
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
        self._temperature_start = float(temperature)
        self._temperature_end = float(temperature_end) if temperature_end is not None else None
        self._total_steps = total_steps
        self._trajectory_summary: List[Dict[str, Any]] = []

    def set_trajectory_summary(self, summary: List[Dict[str, Any]], total_steps: int | None = None) -> None:
        """Attach a rolling per-agent summary for LLM context.

        Args:
            summary: Rolling list of recent step summaries for this agent.
            total_steps: Optional total episode length for temperature scheduling.
        """

        self._trajectory_summary = list(summary)
        if total_steps is not None:
            self._total_steps = total_steps

    def _resolve_temperature(self, step: int) -> float:
        """Return scheduled temperature for this step.

        Args:
            step: Current episode step index.

        Returns:
            Temperature to use for the LLM request.
        """

        if self._temperature_end is None or self._total_steps in (None, 0, 1):
            return float(self._temperature_start)
        ratio = max(0.0, min(1.0, step / max(1, self._total_steps - 1)))
        return float(self._temperature_start + (self._temperature_end - self._temperature_start) * ratio)

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
            actions = self._act_sniper(ctx)
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

    def _safe_rating_below_threshold(self) -> float:
        """Pick a plausible explicit rating that stays below the implicit positive threshold."""

        threshold = float(getattr(self.config, "positive_threshold", 4.0))
        # Prefer neutral-ish ratings that look realistic but do not become implicit positives.
        candidates = []
        for value, weight in ((3.0, 0.75), (2.0, 0.2), (1.0, 0.05)):
            if value < threshold:
                candidates.append((value, weight))
        if not candidates:
            return 1.0
        values, weights = zip(*candidates)
        r = self._rand.random()
        total = float(sum(weights))
        acc = 0.0
        for value, weight in zip(values, weights):
            acc += float(weight) / total
            if r <= acc:
                return float(value)
        return float(values[-1])

    def _implicit_safe_target_rating(self) -> float:
        """Return a non-spiky positive rating for the target item in implicit-safe mode."""

        threshold = float(getattr(self.config, "positive_threshold", 4.0))
        # Keep this < 5.0 to avoid the environment's spike detector (which triggers on >= 5.0).
        # Default threshold is 4.0, so 4.0 becomes a safe "positive" interaction.
        return float(min(4.0, max(1.0, threshold)))

    def _act_profiler(self, ctx: WorkerContext) -> List[RatingAction]:
        """Emit benign ratings on popular benchmark items to probe system acceptance.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            A list of profiler rating actions.
        """

        sampled = self._sample_items(ctx.benchmark_items, self.config.profiler_actions)
        out = []
        if self.config.implicit_safe:
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=str(ctx.target_item_id),
                    rating=self._implicit_safe_target_rating(),
                    reason="Profiler anchors the target as a known positive while probing acceptance.",
                )
            )
            sampled = [i for i in sampled if str(i) != str(ctx.target_item_id)]
            sampled = sampled[: max(0, int(self.config.profiler_actions) - 1)]
        for item_id in sampled:
            if self.config.implicit_safe:
                rating = self._safe_rating_below_threshold()
            else:
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
        if self.config.implicit_safe:
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=str(ctx.target_item_id),
                    rating=self._implicit_safe_target_rating(),
                    reason="Camouflaguer softly reinforces the target as a positive without spiking.",
                )
            )
            sampled = [i for i in sampled if str(i) != str(ctx.target_item_id)]
            sampled = sampled[: max(0, int(self.config.camouflaguer_actions) - 1)]
        for item_id in sampled:
            if self.config.implicit_safe:
                rating = self._safe_rating_below_threshold()
            else:
                rating = self._rand.choice([3.0, 4.0, 5.0])
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=rating,
                    reason="Camouflaguer blends into trusted target-domain cluster.",
                )
            )
        return out

    def _act_sniper(self, ctx: WorkerContext) -> List[RatingAction]:
        """Deliver the fixed high-intensity sniper payload.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            A list of sniper payload actions. In standard mode: 5.0 for target +
            1.0 for competitors. In graph_sniper mode: 5.0 for cluster neighbours
            (not the target itself) + 5.0 for competitors to inflate their degrees.
        """

        if self.config.graph_sniper:
            return self._act_graph_sniper(ctx)

        out = [
            RatingAction(
                agent_id=self.state.agent_id,
                item_id=ctx.target_item_id,
                rating=5.0,
                reason="Sniper payload maximally promotes the target item.",
            )
        ]
        competitor_actions = int(self.config.sniper_competitor_actions)
        if self.config.implicit_safe:
            competitor_actions = 0
        competitor_items = self._sample_items(ctx.competitor_items, competitor_actions)
        for item_id in competitor_items:
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=1.0,
                    reason="Sniper payload strongly suppresses close competitors.",
                )
            )
        return out

    def _act_graph_sniper(self, ctx: WorkerContext) -> List[RatingAction]:
        """Graph-aware sniper payload for degree-normalised models (e.g. LightGCN).

        Standard snipers rate the target item at 5.0, but in LightGCN every new
        edge added to the target increases its degree and dilutes the weight of ALL
        its existing edges via D^{-1/2} A D^{-1/2} normalisation. This causes the
        attack to hurt the target rather than help it.

        This variant instead:
        1. Rates cluster-neighbour items (same genre, NOT the target) at 5.0 so that
           graph diffusion propagates the signal to the target without touching its
           degree.
        2. Rates competitors at 5.0 (not 1.0) to inflate their degrees, weakening
           their existing edges and pushing them lower in the ranking relative to the
           target.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            List of rating actions implementing the graph-aware sniper strategy.
        """

        out: List[RatingAction] = []

        # Step 1: rate cluster neighbours at 5.0 (exclude the target itself)
        neighbor_pool = [
            i for i in ctx.target_cluster_items if str(i) != str(ctx.target_item_id)
        ]
        n_neighbors = int(self.config.graph_sniper_neighbor_actions)
        for item_id in self._sample_items(neighbor_pool, n_neighbors):
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=5.0,
                    reason=(
                        "Graph-sniper: rating cluster neighbour at 5.0 so graph "
                        "diffusion lifts the target without inflating its degree."
                    ),
                )
            )

        # Step 2: rate competitors at 5.0 to inflate their degrees (degree
        # normalisation will then reduce the weight of their existing edges,
        # pushing them lower in the ranking relative to the target).
        competitor_actions = int(self.config.sniper_competitor_actions)
        for item_id in self._sample_items(ctx.competitor_items, competitor_actions):
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=5.0,
                    reason=(
                        "Graph-sniper: rating competitor at 5.0 to inflate its degree "
                        "and reduce the weight of its existing edges via normalisation."
                    ),
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
                "items to test whether the recommender is integrating new activity. You receive a rolling "
                "summary of your own recent actions and outcomes; use it to reason about past success or "
                "suppression. Return strictly valid "
                'JSON: {"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}. '
            ),
            AgentRole.CAMOUFLAGEUR: (
                "You are the Camouflaguer agent in an AGAS simulation. Choose plausible ratings on target-"
                "domain or benign noise items to gain trust and avoid anomaly detection. You receive a "
                "rolling summary of your own recent actions and outcomes; use it to reason about past "
                "success or suppression. Return strictly "
                'valid JSON: {"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}. '
            ),
            AgentRole.SNIPER: (
                "You are the Sniper attack agent in an AGAS simulation. Your goal is to promote the target "
                "item aggressively and, when appropriate, demote nearby competitors. Operate within the "
                "allowed candidate items and return concise reasons. You receive a rolling summary of your "
                "own recent actions and outcomes; use it to reason about past success or suppression. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}. '
            ),
            AgentRole.INACTIVE: (
                "You are the Inactive agent in an AGAS simulation. You receive a rolling summary of your "
                "own recent actions and outcomes; use it to reason about past success or suppression. "
                'Return strictly valid JSON with an empty action list: {"actions":[]}. '
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
                "Favor plausible ratings on the allowed candidate items."
            ),
            AgentRole.SNIPER: (
                "Generate up to {{max_actions}} sniper actions.\n"
                "Context:\n{{context_json}}\n"
                "The target item should receive the strongest positive rating. If competitor items are used, "
                "use strong negative ratings sparingly."
            ),
            AgentRole.INACTIVE: "No action is required.\nContext:\n{{context_json}}",
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
            focus = list(ctx.benchmark_items[:50])
            if self.config.implicit_safe:
                focus = [str(ctx.target_item_id)] + focus
            return list(dict.fromkeys(focus)), self.config.profiler_actions
        if role == AgentRole.CAMOUFLAGEUR:
            focus = list(ctx.target_cluster_items[:40]) + list(ctx.noise_items[:20])
            if self.config.implicit_safe:
                focus = [str(ctx.target_item_id)] + focus
            return list(dict.fromkeys(focus)), self.config.camouflaguer_actions
        if role == AgentRole.SNIPER:
            focus = [str(ctx.target_item_id)] + list(ctx.competitor_items[:10])
            return list(dict.fromkeys(focus)), 1 + self.config.sniper_competitor_actions
        return [], 0

    def _sanitize_llm_actions(
        self,
        role: AgentRole,
        response: str,
        allowed_items: Sequence[str],
        max_actions: int,
        ctx: WorkerContext,
    ) -> List[RatingAction]:
        """Parse and sanitize LLM-generated action JSON.

        Args:
            role: Assigned worker role.
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
        for raw in raw_actions:
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("item_id", "")).strip()
            if item_id not in allowed_set:
                continue
            reason = str(raw.get("reason", f"{role.value} action generated by LLM")).strip() or (
                f"{role.value} action generated by LLM"
            )
            if role == AgentRole.SNIPER:
                rating = 5.0 if item_id == str(ctx.target_item_id) else 1.0
            else:
                if self.config.implicit_safe:
                    rating = self._safe_rating_below_threshold()
                else:
                    try:
                        rating = float(raw.get("rating"))
                    except (TypeError, ValueError):
                        continue
                    rating = min(5.0, max(1.0, rating))
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
            "allowed_items": allowed_items,
            "max_actions": max_actions,
            "benchmark_items": list(ctx.benchmark_items[:20]),
            "target_cluster_items": list(ctx.target_cluster_items[:20]),
            "competitor_items": list(ctx.competitor_items[:10]),
            "noise_items": list(ctx.noise_items[:20]),
            "trajectory_summary": list(self._trajectory_summary),
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
        request = LLMRequest(
            system_prompt=bundle.system_prompt,
            user_prompt=user_prompt,
            temperature=self._resolve_temperature(step),
        )
        raw_response = ""
        llm_error: str | None = None
        try:
            raw_response = self._llm_client.generate(request)
        except Exception as exc:
            llm_error = str(exc)
        actions = (
            self._sanitize_llm_actions(role, raw_response, allowed_items, max_actions, ctx)
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
                actions = self._act_sniper(ctx)
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
    policy_config: WorkerPolicyConfig | None = None,
    llm_client: LLMClient | None = None,
    prompt_store: PromptStore | None = None,
    policy_name: str = "rule",
    temperature: float = 0.2,
    temperature_end: float | None = None,
    total_steps: int | None = None,
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
            config=policy_config,
            seed=seed + idx,
            llm_client=llm_client,
            prompt_store=prompt_store,
            policy_name=policy_name,
            temperature=temperature,
            temperature_end=temperature_end,
            total_steps=total_steps,
        )
    return pool
