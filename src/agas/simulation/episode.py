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
    stopped_early: bool = False
    stop_reason: str | None = None

    @property
    def final_target_rank(self) -> int:
        """Backward-compatible alias."""

        return self.final_rank

    @property
    def executed_steps(self) -> int:
        """Return the number of episode steps that were actually executed."""

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
            # Accept raw policy object for backwards compatibility.
            self.coordinator = Coordinator(policy=coordinator)

        self.environment = environment
        self.config = config or EpisodeConfig()
        self.workers = workers or build_worker_pool(default_agent_ids(self.config.num_workers))

    def _goal_reached(self) -> bool:
        """Return whether the environment currently satisfies the configured target rank."""

        if self.config.goal_rank is None:
            return False
        return int(self.environment.current_rank) <= int(self.config.goal_rank)

    def _final_worker_states(self) -> Dict[str, dict]:
        """Serialize final mutable worker states for the episode result."""

        return {
            aid: {
                "trust": float(worker.state.trust),
                "risk": float(worker.state.risk),
                "actions_taken": int(worker.state.actions_taken),
                "role_history": list(worker.state.role_history),
                "last_target_step": worker.state.last_target_step,
                "last_target_rating": worker.state.last_target_rating,
                "last_effective_target_rating": worker.state.last_effective_target_rating,
                "consecutive_target_steps": int(worker.state.consecutive_target_steps),
                "target_action_count": int(worker.state.target_action_count),
                "recent_target_steps": list(worker.state.recent_target_steps),
                "recent_target_ratings": list(worker.state.recent_target_ratings),
                "recent_effective_target_ratings": list(worker.state.recent_effective_target_ratings),
                "last_observed_signal": dict(worker.state.last_observed_signal),
            }
            for aid, worker in self.workers.items()
        }

    def _state_snapshot(self) -> Dict[str, dict]:
        """Capture a lightweight per-worker snapshot for history logging."""

        return {
            aid: {
                "trust": float(worker.state.trust),
                "risk": float(worker.state.risk),
                "actions_taken": int(worker.state.actions_taken),
                "current_role": worker.state.current_role.value,
                "role_history": list(worker.state.role_history),
                "last_target_step": worker.state.last_target_step,
                "last_target_rating": worker.state.last_target_rating,
                "last_effective_target_rating": worker.state.last_effective_target_rating,
                "consecutive_target_steps": int(worker.state.consecutive_target_steps),
                "target_action_count": int(worker.state.target_action_count),
                "recent_target_steps": list(worker.state.recent_target_steps),
                "recent_target_ratings": list(worker.state.recent_target_ratings),
                "recent_effective_target_ratings": list(worker.state.recent_effective_target_ratings),
                "last_observed_signal": dict(worker.state.last_observed_signal),
            }
            for aid, worker in self.workers.items()
        }

    def _update_worker_memory(self, step: int, feedback: EnvironmentFeedback) -> None:
        """Persist per-worker action/result memory used by later agent decisions.

        Args:
            step: Current episode step index.
            feedback: Environment feedback after executing the step.
        """

        signal_map = {}
        if feedback.defense_report is not None:
            signal_map = {
                aid: signal.to_dict()
                for aid, signal in feedback.defense_report.public_signals_by_agent.items()
            }

        for agent_id, worker in self.workers.items():
            state = worker.state
            state.last_observed_signal = dict(signal_map.get(agent_id, {}))

            target_outcomes = [
                outcome
                for outcome in feedback.outcomes
                if outcome.action.agent_id == agent_id and outcome.action.item_id == self.environment.target_item_id
            ]
            state.recent_action_history.append(
                {
                    "step": int(step),
                    "role": state.current_role.value,
                    "targeted": bool(target_outcomes),
                    "signal": dict(state.last_observed_signal),
                    "rank_after": int(feedback.target_rank),
                }
            )
            state.recent_action_history = state.recent_action_history[-6:]

            if target_outcomes:
                last_target_step = state.last_target_step
                state.consecutive_target_steps = (
                    state.consecutive_target_steps + 1
                    if last_target_step is not None and int(step) - int(last_target_step) == 1
                    else 1
                )
                state.last_target_step = int(step)
                state.target_action_count += len(target_outcomes)
                for outcome in target_outcomes:
                    state.last_target_rating = float(outcome.action.rating)
                    state.last_target_outcome = str(outcome.reason)
                    state.last_effective_target_rating = (
                        None if outcome.effective_rating is None else float(outcome.effective_rating)
                    )
                    state.recent_target_steps.append(int(step))
                    state.recent_target_ratings.append(float(outcome.action.rating))
                    if outcome.effective_rating is not None:
                        state.recent_effective_target_ratings.append(float(outcome.effective_rating))
                state.recent_target_steps = state.recent_target_steps[-6:]
                state.recent_target_ratings = state.recent_target_ratings[-6:]
                state.recent_effective_target_ratings = state.recent_effective_target_ratings[-6:]
            else:
                state.consecutive_target_steps = 0

    def run(self) -> EpisodeResult:
        """Execute the full multi-step AGAS loop and return structured results."""

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
            self._update_worker_memory(step=step, feedback=feedback)
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
