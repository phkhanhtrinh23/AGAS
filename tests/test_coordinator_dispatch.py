"""Tests that the Coordinator dispatches one role per worker each round.

The paper requires the Coordinator to assign exactly one role from
``{PR, SN, CA, IN}`` to every worker in ``U_f`` every round (see
``method_coordinator.tex`` and ``algorithms/agas_end_to_end.tex`` line 12).
This test exercises the legacy :class:`agas.agents.coordinator.Coordinator`
because that is the live dispatch path used by the episode runner.
"""

from __future__ import annotations

from agas.agents.coordinator import Coordinator, RuleBasedCoordinatorPolicy
from agas.agents.messages import AgentRole, CoordinatorObservation
from agas.agents.worker import WorkerState


def test_coordinator_assigns_one_role_per_worker() -> None:
    """Coordinator.assign_roles must return |U_f| assignments, each in
    {PR, SN, CA, IN, DIAGNOSTIC}."""

    agent_ids = [f"agent_{i}" for i in range(1, 5)]
    coordinator = Coordinator(policy=RuleBasedCoordinatorPolicy(agent_order=agent_ids))
    observation = CoordinatorObservation(
        step=3,
        target_item_id="101",
        target_rank=50,
        total_candidates=500,
        target_rank_delta=0,
        total_steps=18,
    )
    worker_states = {
        aid: WorkerState(agent_id=aid, trust=0.4, risk=0.1) for aid in agent_ids
    }

    assignments = coordinator.assign_roles(
        observation=observation, worker_states=worker_states
    )

    # Exactly one assignment per worker.
    assert set(assignments.keys()) == set(agent_ids)
    assert len(assignments) == len(agent_ids)

    # Each assignment role is one of the paper roles (legacy enum also
    # includes DIAGNOSTIC for the S1 probe path).
    allowed = {
        AgentRole.PROFILER,
        AgentRole.SNIPER,
        AgentRole.CAMOUFLAGEUR,
        AgentRole.INACTIVE,
        AgentRole.DIAGNOSTIC,
    }
    for assn in assignments.values():
        assert assn.role in allowed
        assert assn.agent_id in agent_ids
        assert assn.step == observation.step


def test_coordinator_dispatch_scales_with_pool_size() -> None:
    """Eight workers in the pool must produce eight assignments."""

    agent_ids = [f"agent_{i}" for i in range(1, 9)]
    coordinator = Coordinator(policy=RuleBasedCoordinatorPolicy(agent_order=agent_ids))
    observation = CoordinatorObservation(
        step=10,
        target_item_id="101",
        target_rank=12,
        total_candidates=500,
    )
    worker_states = {aid: WorkerState(agent_id=aid) for aid in agent_ids}

    assignments = coordinator.assign_roles(
        observation=observation, worker_states=worker_states
    )
    assert len(assignments) == 8
