"""Coordinator policies for AGAS role assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Protocol, Sequence

from agas.agents.messages import AgentRole, CoordinatorObservation, RoleAssignment
from agas.agents.worker import WorkerState
from agas.llm.providers import LLMClient, LLMRequest


class CoordinatorPolicy(Protocol):
    """Protocol for policies that assign roles to workers."""

    def assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        ...


@dataclass
class RuleBasedCoordinatorPolicy:
    """Rule-based policy that reproduces the described AGAS episode logic."""

    agent_order: Sequence[str] | None = None

    def assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
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

        alerted = [a for a in self.agent_order if a in observation.alerts_by_agent]

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

        trusted = sorted(
            self.agent_order,
            key=lambda aid: worker_states[aid].trust - worker_states[aid].risk,
            reverse=True,
        )
        if trusted:
            assignments[trusted[0]].role = AgentRole.SNIPER
            assignments[trusted[0]].rationale = "Use highest-trust profile for targeted push."
        for aid in trusted[1:3]:
            assignments[aid].role = AgentRole.CAMOUFLAGEUR
            assignments[aid].rationale = "Support trust camouflage and cluster alignment."

        return assignments


@dataclass
class LLMCoordinatorPolicy:
    """LLM-driven coordinator using OpenAI or Ollama backend."""

    client: LLMClient
    agent_order: Sequence[str]

    def assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        prompt = self._build_prompt(observation, worker_states)
        response = self.client.generate(
            LLMRequest(
                system_prompt=(
                    "You are a coordinator for an AGAS simulation. Output strictly valid JSON with keys"
                    " as agent ids and values in {profiler, camouflaguer, sniper, inactive}."
                ),
                user_prompt=prompt,
                temperature=0.1,
            )
        )

        assignments = self._parse_assignments(response, observation.step)
        if assignments:
            return assignments

        fallback = RuleBasedCoordinatorPolicy(agent_order=self.agent_order)
        return fallback.assign(observation, worker_states)

    def _build_prompt(self, observation: CoordinatorObservation, worker_states: Dict[str, WorkerState]) -> str:
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
        self.policy = policy

    def assign_roles(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        return self.policy.assign(observation, worker_states)
