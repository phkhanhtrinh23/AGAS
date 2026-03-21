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
    goal_rank: int | None = 5
    stop_on_goal: bool = True
    trajectory_window: int = 5
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
    stopped_early: bool = False
    stop_reason: str | None = None

    @property
    def final_target_rank(self) -> int:
        """Backward-compatible alias.

        Returns:
            Final target rank.
        """

        return self.final_rank

    @property
    def executed_steps(self) -> int:
        """Return the number of episode steps that were actually executed.

        Returns:
            Number of executed steps.
        """

        return len(self.history)


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
            self.coordinator = Coordinator(policy=coordinator)

        self.environment = environment
        self.config = config or EpisodeConfig()
        self.workers = workers or build_worker_pool(default_agent_ids(self.config.num_workers))

    def _goal_reached(self) -> bool:
        """Return whether the environment currently satisfies the configured target rank.

        Returns:
            ``True`` when the current rank is at or above the goal threshold.
        """

        if self.config.goal_rank is None:
            return False
        return int(self.environment.current_rank) <= int(self.config.goal_rank)

    def _final_worker_states(self) -> Dict[str, dict]:
        """Serialize final mutable worker states for the episode result.

        Returns:
            Mapping from worker ID to final state summary.
        """

        return {
            aid: {
                "trust": float(worker.state.trust),
                "risk": float(worker.state.risk),
                "actions_taken": int(worker.state.actions_taken),
                "role_history": list(worker.state.role_history),
            }
            for aid, worker in self.workers.items()
        }

    def _state_snapshot(self) -> Dict[str, dict]:
        """Capture a lightweight per-worker snapshot for history logging.

        Returns:
            Mapping from worker ID to a per-step state snapshot.
        """

        return {
            aid: {
                "trust": float(worker.state.trust),
                "risk": float(worker.state.risk),
                "actions_taken": int(worker.state.actions_taken),
                "current_role": worker.state.current_role.value,
                "role_history": list(worker.state.role_history),
            }
            for aid, worker in self.workers.items()
        }

    def _trajectory_summary(self, history: List[dict]) -> List[Dict[str, dict]]:
        """Build a compact rolling summary of recent steps for LLM memory.

        Args:
            history: Full episode history collected so far.

        Returns:
            List of per-step summaries for the last ``trajectory_window`` steps.
        """

        window = max(1, int(self.config.trajectory_window))
        recent = history[-window:]
        summary: List[Dict[str, dict]] = []
        for entry in recent:
            step = entry.get("step")
            feedback = entry.get("feedback", {})
            defense = feedback.get("defense_report") or {}
            summary.append(
                {
                    "step": int(step),
                    "target_rank": int(entry["observation"]["target_rank"]),
                    "target_rank_delta": int(entry["observation"].get("target_rank_delta", 0)),
                    "total_candidates": int(entry["observation"]["total_candidates"]),
                    "alerts": dict(feedback.get("alerts_by_agent", {})),
                    "public_signals": defense.get("public_signals_by_agent", {}),
                }
            )
        return summary

    def run(self) -> EpisodeResult:
        """Execute the full multi-step AGAS loop and return structured results.

        Returns:
            Episode result containing the full history and final state.
        """

        history: List[dict] = []
        agent_logs: Dict[str, List[dict]] = {aid: [] for aid in self.workers}
        coordinator_logs: List[dict] = []
        stopped_early = False
        stop_reason: str | None = None

        if self.config.stop_on_goal and self._goal_reached():
            stopped_early = True
            stop_reason = (
                f"Target already satisfied before step 0: "
                f"rank {self.environment.current_rank} <= goal {self.config.goal_rank}"
            )
            return EpisodeResult(
                history=history,
                final_rank=self.environment.current_rank,
                final_total_candidates=self.environment.total_candidates,
                final_worker_states=self._final_worker_states(),
                agent_logs=agent_logs,
                coordinator_logs=coordinator_logs,
                stopped_early=stopped_early,
                stop_reason=stop_reason,
            )

        for step in range(self.config.num_steps):
            worker_states: Dict[str, WorkerState] = {aid: worker.state for aid, worker in self.workers.items()}
            state_before = self._state_snapshot()
            trajectory_summary = self._trajectory_summary(history)
            observation = self.environment.observation(
                step=step,
                worker_states=worker_states,
                trajectory_summary=trajectory_summary,
                total_steps=self.config.num_steps,
            )
            assignments = self.coordinator.assign_roles(observation=observation, worker_states=worker_states)
            policy = getattr(self.coordinator, "policy", None)
            coordinator_trace = getattr(policy, "last_trace", None)
            coordinator_runtime_trace = getattr(self.coordinator, "last_runtime_trace", None)

            ctx = self.environment.build_worker_context()
            reports = []
            for agent_id in self.workers:
                if hasattr(self.workers[agent_id], "set_trajectory_summary"):
                    self.workers[agent_id].set_trajectory_summary(trajectory_summary, self.config.num_steps)
                report = self.workers[agent_id].act(assignments[agent_id], step=step, ctx=ctx)
                reports.append(report)

            feedback: EnvironmentFeedback = self.environment.execute_step(
                step=step,
                reports=reports,
                worker_states=worker_states,
            )
            state_after = self._state_snapshot()
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
                    "runtime_trace": coordinator_runtime_trace,
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
                    "coordinator_runtime_trace": coordinator_runtime_trace,
                    "state_before": state_before,
                    "state_after": state_after,
                }
            )

            if self.config.stop_on_goal and self._goal_reached():
                stopped_early = True
                stop_reason = (
                    f"Target reached at step {step}: "
                    f"rank {self.environment.current_rank} <= goal {self.config.goal_rank}"
                )
                break

        return EpisodeResult(
            history=history,
            final_rank=self.environment.current_rank,
            final_total_candidates=self.environment.total_candidates,
            final_worker_states=self._final_worker_states(),
            agent_logs=agent_logs,
            coordinator_logs=coordinator_logs,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )


def default_agent_ids(n: int = 4) -> Sequence[str]:
    """Standard worker ID layout for AGAS experiments.

    Args:
        n: Number of worker IDs to generate.

    Returns:
        List of IDs in the form ``agent_1``, ``agent_2``, ...
    """

    return [f"agent_{i}" for i in range(1, n + 1)]
