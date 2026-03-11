"""Episode runner for AGAS multi-agent simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from agas.agents.coordinator import Coordinator
from agas.agents.messages import EnvironmentFeedback
from agas.agents.worker import WorkerAgent, WorkerState, build_worker_pool
from agas.simulation.environment import AGASEnvironment


@dataclass
class EpisodeConfig:
    """Episode-level configuration."""

    num_steps: int = 4
    num_workers: int = 4
    # Backward-compatible aliases
    n_steps: int | None = None
    n_workers: int | None = None

    def __post_init__(self) -> None:
        """Hydrate canonical fields from backward-compatible alias arguments."""

        if self.n_steps is not None:
            self.num_steps = self.n_steps
        if self.n_workers is not None:
            self.num_workers = self.n_workers


@dataclass
class EpisodeResult:
    """Structured result returned by episode runner."""

    history: List[dict]
    final_rank: int
    final_total_candidates: int
    final_worker_states: Dict[str, dict]
    agent_logs: Dict[str, List[dict]]
    coordinator_logs: List[dict]

    @property
    def final_target_rank(self) -> int:
        """Backward-compatible alias."""

        return self.final_rank


class AGASEpisodeRunner:
    """Runs coordinator-worker-environment loop for one episode."""

    def __init__(
        self,
        coordinator,
        environment: AGASEnvironment,
        workers: Dict[str, WorkerAgent] | None = None,
        config: EpisodeConfig | None = None,
    ):
        """Bind coordinator, environment, workers, and episode settings.

        Args:
            coordinator: Coordinator object or policy object convertible to
                ``Coordinator``.
            environment: Simulation environment executing actions and defenses.
            workers: Optional worker pool; defaults to newly created workers.
            config: Optional episode configuration.
        """

        if isinstance(coordinator, Coordinator):
            self.coordinator = coordinator
        else:
            # Accept raw policy object for backwards compatibility.
            self.coordinator = Coordinator(policy=coordinator)

        self.environment = environment
        self.config = config or EpisodeConfig()
        self.workers = workers or build_worker_pool(default_agent_ids(self.config.num_workers))

    def run(self) -> EpisodeResult:
        """Execute the full multi-step AGAS loop and return structured results."""

        history: List[dict] = []
        agent_logs: Dict[str, List[dict]] = {aid: [] for aid in self.workers}
        coordinator_logs: List[dict] = []

        for step in range(self.config.num_steps):
            worker_states: Dict[str, WorkerState] = {aid: worker.state for aid, worker in self.workers.items()}
            state_before = {
                aid: {
                    "trust": float(worker.state.trust),
                    "risk": float(worker.state.risk),
                    "actions_taken": int(worker.state.actions_taken),
                    "current_role": worker.state.current_role.value,
                    "role_history": list(worker.state.role_history),
                }
                for aid, worker in self.workers.items()
            }
            observation = self.environment.observation(step=step, worker_states=worker_states)
            assignments = self.coordinator.assign_roles(observation=observation, worker_states=worker_states)
            policy = getattr(self.coordinator, "policy", None)
            coordinator_trace = getattr(policy, "last_trace", None)

            ctx = self.environment.build_worker_context()
            reports = []
            for agent_id in self.workers:
                report = self.workers[agent_id].act(assignments[agent_id], step=step, ctx=ctx)
                reports.append(report)

            feedback: EnvironmentFeedback = self.environment.execute_step(
                step=step,
                reports=reports,
                worker_states=worker_states,
            )
            state_after = {
                aid: {
                    "trust": float(worker.state.trust),
                    "risk": float(worker.state.risk),
                    "actions_taken": int(worker.state.actions_taken),
                    "current_role": worker.state.current_role.value,
                    "role_history": list(worker.state.role_history),
                }
                for aid, worker in self.workers.items()
            }
            outcome_map: Dict[str, List[dict]] = {aid: [] for aid in self.workers}
            for outcome in feedback.outcomes:
                outcome_map.setdefault(outcome.action.agent_id, []).append(outcome.to_dict())

            for report in reports:
                agent_logs.setdefault(report.agent_id, []).append(
                    {
                        "step": step,
                        "assignment": assignments[report.agent_id].to_dict(),
                        "report": report.to_dict(),
                        "outcomes": outcome_map.get(report.agent_id, []),
                        "state_before": state_before[report.agent_id],
                        "state_after": state_after[report.agent_id],
                    }
                )
            coordinator_logs.append(
                {
                    "step": step,
                    "observation": observation.to_dict(),
                    "assignments": {aid: assn.to_dict() for aid, assn in assignments.items()},
                    "trace": coordinator_trace,
                }
            )

            history.append(
                {
                    "step": step,
                    "observation": observation.to_dict(),
                    "assignments": {aid: assn.to_dict() for aid, assn in assignments.items()},
                    "reports": [rep.to_dict() for rep in reports],
                    "feedback": feedback.to_dict(),
                    "coordinator_trace": coordinator_trace,
                    "state_before": state_before,
                    "state_after": state_after,
                }
            )

        final_states = {
            aid: {
                "trust": float(worker.state.trust),
                "risk": float(worker.state.risk),
                "actions_taken": int(worker.state.actions_taken),
                "role_history": list(worker.state.role_history),
            }
            for aid, worker in self.workers.items()
        }

        return EpisodeResult(
            history=history,
            final_rank=self.environment.current_rank,
            final_total_candidates=self.environment.total_candidates,
            final_worker_states=final_states,
            agent_logs=agent_logs,
            coordinator_logs=coordinator_logs,
        )


def default_agent_ids(n: int = 4) -> Sequence[str]:
    """Standard worker ID layout for AGAS experiments.

    Args:
        n: Number of worker IDs to generate.

    Returns:
        List of IDs in the form ``agent_1``, ``agent_2``, ...
    """

    return [f"agent_{i}" for i in range(1, n + 1)]
