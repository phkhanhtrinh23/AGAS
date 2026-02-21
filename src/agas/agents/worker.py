"""Worker agent behaviors for each AGAS role."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Sequence

from agas.agents.messages import AgentRole, RatingAction, RoleAssignment, WorkerActionReport


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


class WorkerAgent:
    """Worker controlled by coordinator role assignments."""

    def __init__(self, state: WorkerState, config: WorkerPolicyConfig | None = None, seed: int = 42):
        self.state = state
        self.config = config or WorkerPolicyConfig()
        self._rand = Random(seed + hash(state.agent_id) % 10_000)

    def act(self, assignment: RoleAssignment, step: int, ctx: WorkerContext) -> WorkerActionReport:
        self.state.current_role = assignment.role
        self.state.role_history.append(assignment.role.value)

        if assignment.role == AgentRole.INACTIVE:
            return WorkerActionReport(step=step, agent_id=self.state.agent_id, role=assignment.role, actions=[])

        if assignment.role == AgentRole.PROFILER:
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
        )

    def _sample_items(self, pool: Sequence[str], n: int) -> List[str]:
        unique_pool = list(dict.fromkeys(pool))
        if not unique_pool:
            return []
        if len(unique_pool) <= n:
            return unique_pool
        return self._rand.sample(unique_pool, n)

    def _act_profiler(self, ctx: WorkerContext) -> List[RatingAction]:
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
        focus_items = list(ctx.target_cluster_items)
        if self.state.risk > 1.0:
            focus_items = list(ctx.noise_items) + focus_items

        sampled = self._sample_items(focus_items, self.config.camouflaguer_actions)
        out = []
        for item_id in sampled:
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
        out = [
            RatingAction(
                agent_id=self.state.agent_id,
                item_id=ctx.target_item_id,
                rating=5.0,
                reason="Sniper payload promotes target item with maximum score.",
            )
        ]
        for item_id in self._sample_items(ctx.competitor_items, self.config.sniper_competitor_actions):
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=1.0,
                    reason="Sniper payload demotes close competitors in target cluster.",
                )
            )
        return out


def build_worker_pool(agent_ids: Sequence[str], seed: int = 42) -> Dict[str, WorkerAgent]:
    """Create worker agents with default policy config."""

    pool = {}
    for idx, agent_id in enumerate(agent_ids):
        state = WorkerState(agent_id=agent_id)
        pool[agent_id] = WorkerAgent(state=state, seed=seed + idx)
    return pool
