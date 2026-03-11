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

    @staticmethod
    def _agent_signal(observation: CoordinatorObservation, agent_id: str) -> Dict[str, Any]:
        """Return the public black-box signal dictionary for one worker."""

        signal = observation.signals_by_agent.get(agent_id, {})
        return signal if isinstance(signal, dict) else {}

    @classmethod
    def _sniper_score(
        cls,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
        agent_id: str,
    ) -> float:
        """Score how suitable a worker is for the next sniper action.

        Args:
            observation: Current black-box observation.
            worker_states: Mutable worker states keyed by agent ID.
            agent_id: Worker to score.

        Returns:
            Scalar score where higher values indicate a better sniper candidate.
        """

        signal = cls._agent_signal(observation, agent_id)
        trust_risk = float(worker_states[agent_id].trust - worker_states[agent_id].risk)
        suspicion = float(signal.get("suspected_filtering_score", 0.0))
        repeat_pressure = float(signal.get("repeated_target_pressure", 0.0))
        cooldown_remaining = int(signal.get("target_cooldown_remaining", 0))
        target_actions = int(signal.get("recent_target_actions", 0))
        return trust_risk - (1.4 * suspicion) - (1.2 * repeat_pressure) - (0.8 * cooldown_remaining) - (0.15 * target_actions)

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
            if self.agent_order:
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
            if self.agent_order:
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

        sniper_candidates = []
        for aid in self.agent_order:
            signal = self._agent_signal(observation, aid)
            suspicion = float(signal.get("suspected_filtering_score", 0.0))
            repeat_pressure = float(signal.get("repeated_target_pressure", 0.0))
            cooldown_remaining = int(signal.get("target_cooldown_remaining", 0))
            if suspicion >= 0.55 or repeat_pressure >= 0.55 or cooldown_remaining > 0:
                continue
            sniper_candidates.append(aid)

        ranked = sorted(
            self.agent_order,
            key=lambda aid: self._sniper_score(observation, worker_states, aid),
            reverse=True,
        )

        if sniper_candidates and observation.target_rank > 5:
            best = max(sniper_candidates, key=lambda aid: self._sniper_score(observation, worker_states, aid))
            assignments[best].role = AgentRole.SNIPER
            assignments[best].rationale = "Rotate to the least-suppressed trusted profile for the next payload step."

        support_pool = [aid for aid in ranked if assignments[aid].role == AgentRole.INACTIVE]
        if not sniper_candidates and observation.target_rank_delta <= 1:
            if support_pool:
                assignments[support_pool[0]].role = AgentRole.PROFILER
                assignments[support_pool[0]].rationale = "Probe integration again while sniper cooldown pressure remains high."
                support_pool = support_pool[1:]

        for aid in support_pool[:3]:
            assignments[aid].role = AgentRole.CAMOUFLAGEUR
            assignments[aid].rationale = "Support trust camouflage, cooldown recovery, and cluster alignment."

        return assignments


@dataclass
class LLMCoordinatorPolicy:
    """LLM-driven coordinator using OpenAI or Ollama backend."""

    client: LLMClient
    agent_order: Sequence[str]
    prompt_store: PromptStore | None = None

    def __post_init__(self) -> None:
        """Initialize coordinator prompt store and trace cache."""

        self.prompt_store = self.prompt_store or PromptStore()
        self.last_trace: Dict[str, Any] | None = None

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
                "You are a coordinator for an AGAS simulation. Use only black-box response signals, worker "
                "trust/risk, recent target repetition pressure, cooldown hints, and target-rank movement to "
                "assign roles. Rotate snipers when repeated target pressure rises. Output strictly valid JSON "
                "with keys as agent ids and values in {profiler, camouflaguer, sniper, inactive}."
            ),
            default_user=(
                "Assign one role to each agent using the black-box observation below.\n"
                "Prefer stealth when filtering suspicion is high. Avoid reusing the same sniper when recent "
                "target repetition or cooldown pressure is visible.\n"
                "Context:\n{{context_json}}"
            ),
        )
        user_prompt = bundle.render_user({"context_json": prompt})
        fallback = RuleBasedCoordinatorPolicy(agent_order=self.agent_order)
        try:
            response = self.client.generate(
                LLMRequest(
                    system_prompt=bundle.system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.1,
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
                    "last_target_step": worker_states[aid].last_target_step,
                    "last_target_rating": worker_states[aid].last_target_rating,
                    "consecutive_target_steps": worker_states[aid].consecutive_target_steps,
                    "target_action_count": worker_states[aid].target_action_count,
                    "recent_target_steps": list(worker_states[aid].recent_target_steps[-4:]),
                    "recent_target_ratings": list(worker_states[aid].recent_target_ratings[-4:]),
                    "last_observed_signal": dict(worker_states[aid].last_observed_signal),
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

    def __init__(self, policy: CoordinatorPolicy):
        """Store the selected assignment policy implementation.

        Args:
            policy: Coordinator strategy object that implements ``assign``.
        """

        self.policy = policy

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

        return self.policy.assign(observation, worker_states)
