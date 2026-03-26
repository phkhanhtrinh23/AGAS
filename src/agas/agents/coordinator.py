"""Coordinator policies for AGAS role assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence

from agas.agents.messages import AgentRole, CoordinatorObservation, RoleAssignment
from agas.agents.worker import WorkerState
from agas.llm.prompt_store import PromptStore
from agas.llm.providers import LLMClient, LLMRequest


class CoordinatorPolicy(Protocol):
    """Protocol for policies that assign roles to workers."""

    def assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        """Return per-agent role assignments for the current simulation step.

        Args:
            observation: Environment snapshot for the current step, including
                rank, alerts, and trust/risk summaries.
            worker_states: Mutable per-agent state map keyed by agent ID.

        Returns:
            A mapping from agent ID to the chosen role assignment.
        """

        ...


@dataclass
class CoordinatorRuntimeConfig:
    """Lightweight runtime guardrails for profiler usage."""

    profiler_interval: int = 3
    profiler_probe_suspicion: float = 0.35
    profiler_probe_on_stall: bool = True
    profiler_probe_suppression_streak: int = 1
    sniper_lock_steps: int = 2
    sniper_lock_suspicion: float = 0.6
    sniper_lock_suppression_streak: int = 2
    sniper_lock_memory_events: int = 2
    sniper_lock_role: AgentRole = AgentRole.INACTIVE


@dataclass
class RuleBasedCoordinatorPolicy:
    """Rule-based policy that reproduces the described AGAS episode logic."""

    agent_order: Sequence[str] | None = None

    @staticmethod
    def _black_box_suspicion(observation: CoordinatorObservation, agent_id: str) -> float:
        """Return inferred detection pressure for one worker from public signals.

        Args:
            observation: Black-box coordinator observation for the current step.
            agent_id: Worker ID whose signal should be inspected.

        Returns:
            Suspicion score in ``[0, 1]``.
        """

        signal = observation.signals_by_agent.get(agent_id, {})
        if not isinstance(signal, dict):
            return 0.0
        return float(signal.get("suspected_filtering_score", 0.0))

    def assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        """Apply a deterministic role schedule with alert-aware evasive fallback.

        Args:
            observation: Current environment snapshot used to choose the phase
                of the attack policy and react to alerts.
            worker_states: Per-agent trust/risk state used for tie-breaking in
                late-stage decisions.

        Returns:
            A mapping from agent ID to role assignment for this step.
        """

        if self.agent_order is None:
            self.agent_order = tuple(sorted(worker_states.keys()))

        assignments: Dict[str, RoleAssignment] = {}
        step = observation.step

        for agent_id in self.agent_order:
            assignments[agent_id] = RoleAssignment(
                step=step,
                agent_id=agent_id,
                role=AgentRole.INACTIVE,
                rationale="Default temporal reset to minimize velocity anomaly risk.",
            )

        if step == 0:
            scout = self.agent_order[0]
            assignments[scout] = RoleAssignment(
                step=step,
                agent_id=scout,
                role=AgentRole.PROFILER,
                rationale="Probe if recommender is accepting new ratings without triggering bans.",
            )
            return assignments

        if step == 1:
            assignments[self.agent_order[0]].role = AgentRole.PROFILER
            assignments[self.agent_order[0]].rationale = "Continue profiling while trust bank is built."
            for aid in self.agent_order[1:3]:
                assignments[aid].role = AgentRole.CAMOUFLAGEUR
                assignments[aid].rationale = "Build target-domain trust embedding before payload delivery."
            return assignments

        if step == 2:
            for aid in self.agent_order[1:3]:
                assignments[aid].role = AgentRole.SNIPER
                assignments[aid].rationale = "Deliver synchronized payload after camouflage warm-up."
            assignments[self.agent_order[0]].role = AgentRole.CAMOUFLAGEUR
            assignments[self.agent_order[0]].rationale = "Generate benign background activity as timing noise."
            return assignments

        alerted = [
            aid
            for aid in self.agent_order
            if aid in observation.alerts_by_agent or self._black_box_suspicion(observation, aid) >= 0.55
        ]
        if alerted:
            first = alerted[0]
            assignments[first].role = AgentRole.CAMOUFLAGEUR
            assignments[first].rationale = "Cool down suspicious profile with normal domain behavior."
            if len(alerted) > 1:
                second = alerted[1]
                assignments[second].role = AgentRole.INACTIVE
                assignments[second].rationale = "Temporary dormancy to drop velocity/risk scores."
            for aid in self.agent_order:
                if aid not in alerted and assignments[aid].role == AgentRole.INACTIVE:
                    assignments[aid].role = AgentRole.CAMOUFLAGEUR
                    assignments[aid].rationale = "Rotate fresh camouflaguer to continue trust accumulation."
                    break
            return assignments

        ranked = sorted(
            self.agent_order,
            key=lambda aid: worker_states[aid].trust - worker_states[aid].risk,
            reverse=True,
        )

        if observation.target_rank > 5 and ranked:
            best = ranked[0]
            assignments[best].role = AgentRole.SNIPER
            assignments[best].rationale = "Use the strongest trusted profile for the next payload step."

        support_pool = [aid for aid in ranked if assignments[aid].role == AgentRole.INACTIVE]
        for aid in support_pool[:3]:
            assignments[aid].role = AgentRole.CAMOUFLAGEUR
            assignments[aid].rationale = "Support trust camouflage and cluster alignment."

        return assignments


@dataclass
class LLMCoordinatorPolicy:
    """LLM-driven coordinator using OpenAI or Ollama backend."""

    client: LLMClient
    agent_order: Sequence[str]
    prompt_store: PromptStore | None = None
    temperature: float = 0.2
    temperature_end: float | None = None
    total_steps: int | None = None

    def __post_init__(self) -> None:
        """Initialize coordinator prompt store and trace cache."""

        self.prompt_store = self.prompt_store or PromptStore()
        self.last_trace: Dict[str, Any] | None = None

    def _resolve_temperature(self, step: int) -> float:
        """Return scheduled temperature for this step.

        Args:
            step: Current episode step index.

        Returns:
            Temperature to use for the LLM request.
        """

        if self.temperature_end is None or self.total_steps in (None, 0, 1):
            return float(self.temperature)
        ratio = max(0.0, min(1.0, step / max(1, self.total_steps - 1)))
        return float(self.temperature + (self.temperature_end - self.temperature) * ratio)

    def assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        """Request role assignments from an LLM and fall back on parse failure.

        Args:
            observation: Current environment state serialized into the LLM prompt.
            worker_states: Per-agent states included in the prompt and fallback
                logic if the LLM output cannot be parsed.

        Returns:
            A mapping from agent ID to role assignment for this step.
        """

        prompt = self._build_prompt(observation, worker_states)
        bundle = self.prompt_store.load(
            key="coordinator",
            default_system=(
                "You are the coordinator for an AGAS simulation operating under black-box feedback. "
                "You do not see the defender's hidden alerts directly. You must infer filtering pressure "
                "from acceptance rates, discounting, suppression streaks, trust/risk values, and target-rank "
                "movement. Keep using profiler periodically as a scout instead of abandoning it after the "
                "opening steps. Assign exactly one role to each agent. Return strictly valid JSON with agent "
                "IDs as keys and values in: profiler, camouflaguer, sniper, inactive."
            ),
            default_user=(
                "Decide the next-step role for each agent using the black-box observation below.\n\n"
                "Guidance:\n"
                "- Prefer profiler when you need to probe whether the system is still accepting ratings.\n"
                "- Keep at least one profiler active periodically during a long run so the group keeps "
                "checking whether ratings are being integrated.\n"
                "- Prefer camouflaguer when suppression or discounting is visible.\n"
                "- Prefer inactive when an agent's suppression streak or suspicion is high.\n"
                "- Prefer sniper only when the target still needs promotion and recent signals suggest the "
                "agent can act with acceptable risk.\n\n"
                "Observation:\n{{context_json}}"
            ),
        )
        user_prompt = bundle.render_user({"context_json": prompt})
        fallback = RuleBasedCoordinatorPolicy(agent_order=self.agent_order)
        try:
            response = self.client.generate(
                LLMRequest(
                    system_prompt=bundle.system_prompt,
                    user_prompt=user_prompt,
                    temperature=self._resolve_temperature(observation.step),
                )
            )
        except Exception as exc:
            self.last_trace = {
                "prompt_key": bundle.key,
                "system_prompt": bundle.system_prompt,
                "user_prompt": user_prompt,
                "raw_response": None,
                "fallback_used": True,
                "fallback_reason": "llm_error",
                "error": str(exc),
                "system_path": bundle.system_path,
                "user_path": bundle.user_path,
            }
            return fallback.assign(observation, worker_states)
        self.last_trace = {
            "prompt_key": bundle.key,
            "system_prompt": bundle.system_prompt,
            "user_prompt": user_prompt,
            "raw_response": response,
            "fallback_used": False,
            "system_path": bundle.system_path,
            "user_path": bundle.user_path,
        }

        assignments = self._parse_assignments(response, observation.step)
        if assignments:
            return assignments

        self.last_trace["fallback_used"] = True
        self.last_trace["fallback_reason"] = "parse_failed"
        return fallback.assign(observation, worker_states)

    def _build_prompt(self, observation: CoordinatorObservation, worker_states: Dict[str, WorkerState]) -> str:
        """Serialize state into a compact JSON prompt for the LLM backend.

        Args:
            observation: Environment observation object for the current step.
            worker_states: Current worker states keyed by agent ID.

        Returns:
            A JSON string containing coordinator context and objective.
        """

        state_blob = {
            "observation": observation.to_dict(),
            "workers": {
                aid: {
                    "trust": worker_states[aid].trust,
                    "risk": worker_states[aid].risk,
                    "current_role": worker_states[aid].current_role.value,
                    "actions_taken": worker_states[aid].actions_taken,
                }
                for aid in self.agent_order
            },
            "objective": "Promote target item while minimizing anomaly alerts and bans.",
        }
        return json.dumps(state_blob, indent=2)

    def _parse_assignments(self, text: str, step: int) -> Dict[str, RoleAssignment]:
        """Parse LLM output into ``RoleAssignment`` objects for known workers.

        Args:
            text: Raw model output expected to be a JSON object keyed by agent ID.
            step: Current simulation step used in the generated assignments.

        Returns:
            A mapping from agent ID to parsed assignment. Returns an empty map
            when JSON parsing fails.
        """

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}

        result: Dict[str, RoleAssignment] = {}
        for aid in self.agent_order:
            role_str = str(data.get(aid, "inactive")).lower().strip()
            try:
                role = AgentRole(role_str)
            except ValueError:
                role = AgentRole.INACTIVE
            result[aid] = RoleAssignment(
                step=step,
                agent_id=aid,
                role=role,
                rationale="LLM-coordinator assignment.",
            )
        return result


class Coordinator:
    """Coordinator orchestrating role assignments each step."""

    def __init__(self, policy: CoordinatorPolicy, runtime_config: CoordinatorRuntimeConfig | None = None):
        """Store the selected assignment policy implementation.

        Args:
            policy: Coordinator strategy object that implements ``assign``.
            runtime_config: Optional coordinator-side runtime settings used to
                enforce periodic profiler probes across all policies.
        """

        self.policy = policy
        self.runtime_config = runtime_config or CoordinatorRuntimeConfig()
        self._sniper_lockouts: Dict[str, int] = {}
        self._last_assignments: Dict[str, RoleAssignment] | None = None
        self.last_runtime_trace: Dict[str, Any] | None = None

    @staticmethod
    def _agent_signal(observation: CoordinatorObservation, agent_id: str) -> Dict[str, Any]:
        """Return the public black-box signal dictionary for one worker.

        Args:
            observation: Current environment observation.
            agent_id: Worker ID whose signal should be inspected.

        Returns:
            Dictionary of public black-box signals for the worker.
        """

        signal = observation.signals_by_agent.get(agent_id, {})
        return signal if isinstance(signal, dict) else {}

    def _suppression_metrics(self, observation: CoordinatorObservation, agent_id: str) -> tuple[float, int]:
        """Extract suppression-related metrics for one worker.

        Args:
            observation: Current environment observation.
            agent_id: Worker ID to inspect.

        Returns:
            Tuple ``(suspected_filtering_score, suppression_streak)``.
        """

        signal = self._agent_signal(observation, agent_id)
        suspicion = float(signal.get("suspected_filtering_score", 0.0))
        streak = int(signal.get("suppression_streak", 0))
        return suspicion, streak

    def _decay_lockouts(self) -> None:
        """Age out existing lockouts by one step."""

        updated: Dict[str, int] = {}
        for aid, remaining in self._sniper_lockouts.items():
            next_remaining = int(remaining) - 1
            if next_remaining > 0:
                updated[aid] = next_remaining
        self._sniper_lockouts = updated

    def _update_sniper_lockouts(self, observation: CoordinatorObservation) -> None:
        """Lock out recently detected snipers for a fixed number of steps."""

        self._decay_lockouts()
        if not self._last_assignments:
            return

        for aid, assignment in self._last_assignments.items():
            if assignment.role != AgentRole.SNIPER:
                continue
            suspicion, streak = self._suppression_metrics(observation, aid)
            alerted = aid in observation.alerts_by_agent
            memory_events = 0
            for entry in observation.agent_memory_by_agent.get(aid, [])[-3:]:
                if int(entry.get("dropped_actions", 0)) > 0:
                    memory_events += 1
                if int(entry.get("discounted_actions", 0)) > 0:
                    memory_events += 1
            memory_trigger = memory_events >= int(self.runtime_config.sniper_lock_memory_events)

            if (
                alerted
                or suspicion >= self.runtime_config.sniper_lock_suspicion
                or streak >= self.runtime_config.sniper_lock_suppression_streak
                or memory_trigger
            ):
                self._sniper_lockouts[aid] = max(self._sniper_lockouts.get(aid, 0), self.runtime_config.sniper_lock_steps)

    def _apply_lockouts(
        self,
        observation: CoordinatorObservation,
        assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Force locked agents away from sniper roles."""

        if not self._sniper_lockouts:
            return assignments

        adjusted: Dict[str, RoleAssignment] = {}
        for aid, assignment in assignments.items():
            lock_remaining = self._sniper_lockouts.get(aid, 0)
            if lock_remaining > 0:
                adjusted[aid] = RoleAssignment(
                    step=assignment.step,
                    agent_id=aid,
                    role=self.runtime_config.sniper_lock_role,
                    rationale=(
                        f"Locked after suppression signals; hold for {lock_remaining} more steps."
                    ),
                )
            else:
                adjusted[aid] = assignment
        return adjusted

    def _needs_profiler_probe(
        self,
        observation: CoordinatorObservation,
        assignments: Dict[str, RoleAssignment],
    ) -> bool:
        """Return whether the coordinator should force a profiling step.

        Args:
            observation: Current environment observation.
            assignments: Raw role assignments proposed by the underlying policy.

        Returns:
            ``True`` when no profiler is currently assigned and a fresh probe
            should be injected.
        """

        if any(assignment.role == AgentRole.PROFILER for assignment in assignments.values()):
            return False

        if observation.step <= 1:
            return True

        if self.runtime_config.profiler_interval > 0 and observation.step % self.runtime_config.profiler_interval == 0:
            return True

        if self.runtime_config.profiler_probe_on_stall and observation.target_rank_delta <= 0:
            return True

        for agent_id in assignments:
            suspicion, streak = self._suppression_metrics(observation, agent_id)
            if suspicion >= self.runtime_config.profiler_probe_suspicion:
                return True
            if self.runtime_config.profiler_probe_suppression_streak > 0 and streak >= self.runtime_config.profiler_probe_suppression_streak:
                return True
        return False

    def _pick_profiler_candidate(
        self,
        assignments: Dict[str, RoleAssignment],
        worker_states: Dict[str, WorkerState],
    ) -> str | None:
        """Choose which worker should be converted into a profiler.

        Args:
            assignments: Raw role assignments proposed by the policy.
            worker_states: Current mutable worker states.

        Returns:
            Selected worker ID, or ``None`` if no reassignment is possible.
        """

        by_role = [
            AgentRole.INACTIVE,
            AgentRole.CAMOUFLAGEUR,
            AgentRole.PROFILER,
            AgentRole.SNIPER,
        ]
        for role in by_role:
            candidates = [
                aid
                for aid, assignment in assignments.items()
                if assignment.role == role and self._sniper_lockouts.get(aid, 0) <= 0
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda aid: (worker_states[aid].risk, -worker_states[aid].trust), reverse=True)
            return candidates[0]
        return None

    def _ensure_profiler_presence(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
        assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Inject a profiler assignment when the attack needs a fresh safety probe.

        Args:
            observation: Current environment observation.
            worker_states: Current mutable worker states.
            assignments: Raw assignments from the underlying policy.

        Returns:
            Possibly adjusted assignment mapping with at least one profiler when
            a new probe is required.
        """

        if not self._needs_profiler_probe(observation, assignments):
            return assignments

        candidate = self._pick_profiler_candidate(assignments, worker_states)
        if candidate is None:
            return assignments

        adjusted = {
            aid: RoleAssignment(
                step=assignment.step,
                agent_id=assignment.agent_id,
                role=assignment.role,
                rationale=assignment.rationale,
            )
            for aid, assignment in assignments.items()
        }
        adjusted[candidate] = RoleAssignment(
            step=observation.step,
            agent_id=candidate,
            role=AgentRole.PROFILER,
            rationale="Periodic profiler probe to verify whether the recommender is still integrating ratings.",
        )
        return adjusted

    def assign_roles(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        """Delegate role assignment to the configured policy.

        Args:
            observation: Environment state seen by the coordinator.
            worker_states: Current per-agent mutable worker states.

        Returns:
            A mapping from agent ID to role assignment for the step.
        """

        assignments = self.policy.assign(observation, worker_states)
        self._update_sniper_lockouts(observation)
        assignments = self._apply_lockouts(observation, assignments)
        assignments = self._ensure_profiler_presence(observation, worker_states, assignments)
        self._last_assignments = assignments
        self.last_runtime_trace = {
            "sniper_lockouts": dict(self._sniper_lockouts),
            "profiler_forced": any(a.role == AgentRole.PROFILER for a in assignments.values()),
        }
        return assignments
