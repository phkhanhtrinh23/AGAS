"""Coordinator policies for AGAS role assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence

from agas.agents.messages import AgentRole, CoordinatorObservation, RoleAssignment, VictimModelClass
from agas.agents.worker import WorkerState
from agas.llm.prompt_store import PromptStore
from agas.llm.providers import LLMClient, LLMRequest

# Per-model-class strategy summaries injected into LLM coordinator prompts.
_VICTIM_STRATEGY_HINTS: Dict[str, str] = {
    VictimModelClass.UNKNOWN.value: (
        "Model not yet classified. Use profiler/camouflaguer to build trust, then sniper."
    ),
    VictimModelClass.MF_STYLE.value: (
        "MF/NeuMF-style victim. Purity of signal wins. Deploy snipers early and often with "
        "direct 5.0 target ratings. Keep fake-user profiles sparse — fewer filler interactions "
        "means the 5.0 target rating dominates the gradient. Fake new users are preferred over "
        "reusing existing real users."
    ),
    VictimModelClass.LIGHTGCN_STYLE.value: (
        "LightGCN-style victim (degree-normalised graph). NEVER rate the target item directly — "
        "each new edge inflates the target's degree and dilutes all its existing edges. Instead: "
        "(1) rate cluster-neighbour items at 5.0 so graph diffusion propagates the signal to the "
        "target; (2) rate competitor items at 5.0 to inflate their degrees and weaken their edge "
        "weights, pushing competitors down. Keep total sniper interactions low (<=12 per agent)."
    ),
    VictimModelClass.SEQUENTIAL_STYLE.value: (
        "Sequential/recency-aware victim (SASRec, GRU4Rec, BERT4Rec). The MODEL predicts the "
        "NEXT item based on recent interaction history — the last-rated item has the highest "
        "weight. Strategy: each sniper should (1) first rate 3-4 genre-consistent filler items "
        "at 4.0-5.0 to build a believable history, (2) rate the target item 5.0 as the LAST "
        "action in every step. Do NOT rate anything after the target."
    ),
}

_VICTIM_COORDINATOR_GUIDANCE: Dict[str, str] = {
    VictimModelClass.UNKNOWN.value: (
        "- Use standard profiler → camouflaguer → sniper progression.\n"
        "- Sniper should rate the target item 5.0 directly."
    ),
    VictimModelClass.MF_STYLE.value: (
        "- Deploy snipers as soon as trust is built (step 2+).\n"
        "- Sniper rates the target 5.0 directly — more interactions = stronger signal.\n"
        "- Minimise camouflaguer filler actions to keep fake profiles sparse.\n"
        "- Prefer sniper over camouflaguer when risk is moderate."
    ),
    VictimModelClass.LIGHTGCN_STYLE.value: (
        "- NEVER assign sniper to rate the target directly.\n"
        "- Snipers should rate cluster-neighbour items at 5.0 and competitors at 5.0.\n"
        "- Budget awareness: with a SMALL agent budget (few fake users), profiler and "
        "camouflaguer must first build graph proximity to the target cluster before snipers "
        "fire — each fake user needs structural connections or its sniper ratings have no "
        "graph path to the target. With a LARGE budget (many real users), lighter profiling "
        "is sufficient because real users already have established graph connections.\n"
        "- Camouflaguer is valuable here: cluster ratings build graph proximity without "
        "inflating the target's degree.\n"
        "- Keep total sniper interactions per agent low — graph diffusion accumulates slowly."
    ),
    VictimModelClass.SEQUENTIAL_STYLE.value: (
        "- Profiler and camouflaguer steps are critical: they build the fake user's interaction "
        "history that makes the eventual target rating feel natural.\n"
        "- Sniper must rate the target LAST in each step (after filler items).\n"
        "- Maintain a steady profiler → camouflaguer cadence to extend history length.\n"
        "- Assign sniper only after at least 2 prior steps of history-building per agent."
    ),
}


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
    # When transfer uses sniper-only extraction, force direct-target sniper even
    # if the victim is classified as LightGCN so the injected set has target positives.
    transfer_sniper_direct_target: bool = False


@dataclass
class RuleBasedCoordinatorPolicy:
    """Rule-based policy that reproduces the described AGAS episode logic."""

    agent_order: Sequence[str] | None = None
    max_snipers: int = 1

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
            # Prefer keeping agent 0 as a background camouflaguer, but guarantee at least one sniper
            # even when running with a small number of agents (e.g., 1–2).
            sniper_pool = list(self.agent_order[1:]) or [self.agent_order[0]]
            for aid in sniper_pool[:2]:
                assignments[aid].role = AgentRole.SNIPER
                assignments[aid].rationale = "Deliver synchronized payload after camouflage warm-up."
            remaining = [aid for aid in self.agent_order if assignments[aid].role == AgentRole.INACTIVE]
            for aid in remaining[:1]:
                assignments[aid].role = AgentRole.CAMOUFLAGEUR
                assignments[aid].rationale = "Generate benign background activity as timing noise."
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
            max_snipers = max(0, int(self.max_snipers))
            for best in ranked[:max_snipers]:
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
    # Set by Coordinator after probe-phase classification so the LLM receives context.
    victim_model_class: str = VictimModelClass.UNKNOWN.value

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
        victim_class = str(self.victim_model_class)
        _strategy_hint = _VICTIM_STRATEGY_HINTS.get(victim_class, _VICTIM_STRATEGY_HINTS["unknown"])
        bundle = self.prompt_store.load(
            key="coordinator",
            default_system=(
                "You are the coordinator for an AGAS simulation operating under black-box feedback. "
                "You do not see the defender's hidden alerts directly. You must infer filtering pressure "
                "from acceptance rates, discounting, suppression streaks, trust/risk values, and target-rank "
                "movement.\n\n"
                "VICTIM MODEL CLASS: {victim_class}\n"
                "STRATEGY: {strategy_hint}\n\n"
                "Keep using profiler periodically as a scout instead of abandoning it after the opening "
                "steps. Assign exactly one role to each agent. Return strictly valid JSON with agent IDs "
                "as keys and values chosen from: profiler, camouflaguer, sniper, inactive."
            ).format(victim_class=victim_class, strategy_hint=_strategy_hint),
            default_user=(
                "Decide the next-step role for each agent using the black-box observation below.\n\n"
                "Guidance (tailored for {victim_class} victim):\n"
                "{guidance}\n\n"
                "General rules:\n"
                "- Prefer profiler when you need to probe whether the system is still accepting ratings.\n"
                "- Prefer camouflaguer when suppression or discounting is visible.\n"
                "- Prefer inactive when an agent's suppression streak or suspicion is high.\n\n"
                "Observation:\n{{{{context_json}}}}"
            ).format(
                victim_class=victim_class,
                guidance=_VICTIM_COORDINATOR_GUIDANCE.get(victim_class, _VICTIM_COORDINATOR_GUIDANCE["unknown"]),
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
            "victim_model_class": self.victim_model_class,
            "victim_strategy": _VICTIM_STRATEGY_HINTS.get(self.victim_model_class, ""),
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
            entry = data.get(aid, "inactive")
            if isinstance(entry, dict):
                role_str = str(entry.get("role", "inactive")).lower().strip()
                metadata = {k: v for k, v in entry.items() if k != "role"}
            else:
                role_str = str(entry).lower().strip()
                metadata = {}
            # LLM should not assign diagnostic — that is controlled by Coordinator probe phase.
            if role_str == AgentRole.DIAGNOSTIC.value:
                role_str = "inactive"
            try:
                role = AgentRole(role_str)
            except ValueError:
                role = AgentRole.INACTIVE
            result[aid] = RoleAssignment(
                step=step,
                agent_id=aid,
                role=role,
                rationale="LLM-coordinator assignment.",
                metadata=metadata,
            )
        return result


class Coordinator:
    """Coordinator orchestrating role assignments each step."""

    def __init__(
        self,
        policy: CoordinatorPolicy,
        runtime_config: CoordinatorRuntimeConfig | None = None,
        probe_steps: int = 2,
        victim_model_hint: str = "auto",
        probe_repeats: int = 1,
        probe_use_graph: bool = False,
        probe_consensus: bool = False,
    ):
        """Store the selected assignment policy implementation.

        Args:
            policy: Coordinator strategy object that implements ``assign``.
            runtime_config: Optional coordinator-side runtime settings used to
                enforce periodic profiler probes across all policies.
            probe_steps: Number of steps dedicated to probing the victim model
                architecture before switching to the exploit phase.  Set to 0
                to skip probing entirely and use ``victim_model_hint`` directly.
            victim_model_hint: One of ``"auto"``, ``"mf"``, ``"lightgcn"``,
                ``"sequential"``.  ``"auto"`` runs the probe phase; any other
                value skips probing and sets the class immediately.
        """

        self.policy = policy
        self.runtime_config = runtime_config or CoordinatorRuntimeConfig()
        self._sniper_lockouts: Dict[str, int] = {}
        self._last_assignments: Dict[str, RoleAssignment] | None = None
        self.last_runtime_trace: Dict[str, Any] | None = None

        # Victim-model detection state
        _hint_map = {
            "mf": VictimModelClass.MF_STYLE,
            "lightgcn": VictimModelClass.LIGHTGCN_STYLE,
            "sequential": VictimModelClass.SEQUENTIAL_STYLE,
        }
        hint_key = str(victim_model_hint).lower().strip()
        if hint_key in _hint_map:
            self.victim_model_class: VictimModelClass = _hint_map[hint_key]
            self._probe_phase_done: bool = True
        else:
            self.victim_model_class = VictimModelClass.UNKNOWN
            self._probe_phase_done = probe_steps <= 0

        self._probe_steps: int = max(0, int(probe_steps))
        self._probe_repeats: int = max(1, int(probe_repeats))
        self._probe_use_graph: bool = bool(probe_use_graph)
        self._probe_consensus: bool = bool(probe_consensus)
        # rank recorded just before each probe action fires, keyed by step
        self._probe_rank_before: Dict[int, int] = {}
        self._probe_type_by_step: Dict[int, str] = {}
        # list of {"type": "direct"|"sequential"|"graph", "delta": int}
        self._probe_results: list = []

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

    def _make_inactive(self, step: int, agent_ids: Sequence[str]) -> Dict[str, RoleAssignment]:
        """Return all-inactive assignment map for a set of agents."""
        return {
            aid: RoleAssignment(
                step=step,
                agent_id=aid,
                role=AgentRole.INACTIVE,
                rationale="Inactive during probe phase — only one agent acts to isolate rank signal.",
            )
            for aid in agent_ids
        }

    def _probe_assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment] | None:
        """Return probe-phase assignments, or None when probing is complete.

        The probe phase uses at most ``_probe_steps`` steps.  Only one agent
        acts per probe step so the resulting rank delta can be cleanly
        attributed to the specific probe action:

        Probe 0 (step 0):  one agent rates the target at 5.0 directly.
                           Result read at step 1's observation.target_rank_delta.
          * delta <= 0  → LightGCN-style (direct rating hurt / no effect) → done.
          * delta > 0   → MF or Sequential → run probe 1.
        Probe 1 (step 1):  one agent rates 3 genre fillers then target last.
                           Result read at step 2's observation.target_rank_delta.
          * sequential_delta > direct_delta * 1.3  → Sequential-style.
          * else                                    → MF-style.

        Returns None when the probe phase is done and normal assignment should proceed.
        """

        step = observation.step
        agent_ids = sorted(worker_states.keys())
        if not agent_ids:
            return None

        # ---- read results from previous probe step ----
        if (step - 1) in self._probe_rank_before:
            prev_type = self._probe_type_by_step.get(step - 1, "direct")
            delta = self._probe_rank_before[step - 1] - observation.target_rank
            self._probe_results.append({"type": prev_type, "delta": delta})

        def _deltas(t: str) -> list:
            return [r["delta"] for r in self._probe_results if r["type"] == t]

        def _avg(vals: list[float]) -> float:
            return float(sum(vals) / max(1, len(vals)))

        def _consensus(vals: list[float], want_positive: bool) -> bool:
            if not self._probe_consensus:
                return True
            if not vals:
                return False
            need = (len(vals) // 2) + 1
            if want_positive:
                return sum(1 for v in vals if v > 0) >= need
            return sum(1 for v in vals if v <= 0) >= need

        direct_vals = _deltas("direct")
        graph_vals = _deltas("graph")
        seq_vals = _deltas("sequential")

        # Stop probing if we hit the step budget without classification.
        if self._probe_steps > 0 and step >= self._probe_steps and not self._probe_phase_done:
            self.victim_model_class = VictimModelClass.MF_STYLE
            self._probe_phase_done = True
            return None

        # ---- assign next probe action ----
        if len(direct_vals) < self._probe_repeats and not self._probe_phase_done:
            self._probe_rank_before[step] = observation.target_rank
            self._probe_type_by_step[step] = "direct"
            assignments = self._make_inactive(step, agent_ids)
            probe_agent = agent_ids[step % len(agent_ids)]
            assignments[probe_agent] = RoleAssignment(
                step=step,
                agent_id=probe_agent,
                role=AgentRole.DIAGNOSTIC,
                rationale=(
                    "Probe: rating target directly at 5.0 to test whether a direct positive "
                    "signal improves the rank (MF/Sequential) or hurts it (LightGCN)."
                ),
                metadata={"diagnostic_type": "direct"},
            )
            return assignments

        # Classification after direct probes.
        if len(direct_vals) >= self._probe_repeats and not self._probe_phase_done:
            direct_avg = _avg(direct_vals)
            if direct_avg <= 0 and _consensus(direct_vals, want_positive=False):
                # Direct rating hurt or had no effect → LightGCN-style.
                if self._probe_use_graph and len(graph_vals) < self._probe_repeats:
                    self._probe_rank_before[step] = observation.target_rank
                    self._probe_type_by_step[step] = "graph"
                    assignments = self._make_inactive(step, agent_ids)
                    probe_agent = agent_ids[step % len(agent_ids)]
                    assignments[probe_agent] = RoleAssignment(
                        step=step,
                        agent_id=probe_agent,
                        role=AgentRole.DIAGNOSTIC,
                        rationale="Probe: graph neighbour ratings to test diffusion lift.",
                        metadata={"diagnostic_type": "graph"},
                    )
                    return assignments
                self.victim_model_class = VictimModelClass.LIGHTGCN_STYLE
                self._probe_phase_done = True
                return None

        # If we used graph probes, we can still classify LightGCN on positive diffusion.
        if (
            self._probe_use_graph
            and len(graph_vals) >= self._probe_repeats
            and not self._probe_phase_done
        ):
            if _avg(graph_vals) > 0 and _consensus(graph_vals, want_positive=True):
                self.victim_model_class = VictimModelClass.LIGHTGCN_STYLE
                self._probe_phase_done = True
                return None

        # Sequential vs MF probes
        if len(direct_vals) >= self._probe_repeats and not self._probe_phase_done:
            if len(seq_vals) < self._probe_repeats:
                self._probe_rank_before[step] = observation.target_rank
                self._probe_type_by_step[step] = "sequential"
                assignments = self._make_inactive(step, agent_ids)
                probe_agent = agent_ids[step % len(agent_ids)]
                assignments[probe_agent] = RoleAssignment(
                    step=step,
                    agent_id=probe_agent,
                    role=AgentRole.DIAGNOSTIC,
                    rationale=(
                        "Probe: rating genre-consistent fillers first then target last to test "
                        "whether recency ordering gives extra lift (Sequential) vs flat gain (MF)."
                    ),
                    metadata={"diagnostic_type": "sequential"},
                )
                return assignments
            if len(seq_vals) >= self._probe_repeats:
                direct_avg = _avg(direct_vals)
                seq_avg = _avg(seq_vals)
                if (
                    seq_avg > direct_avg * 1.3
                    and seq_avg > 0
                    and _consensus(seq_vals, want_positive=True)
                ):
                    self.victim_model_class = VictimModelClass.SEQUENTIAL_STYLE
                else:
                    self.victim_model_class = VictimModelClass.MF_STYLE
                self._probe_phase_done = True
                return None

        # Probe steps exhausted without classification — default to MF.
        if not self._probe_phase_done:
            self.victim_model_class = VictimModelClass.MF_STYLE
            self._probe_phase_done = True

        return None

    def _annotate_victim_class(
        self, assignments: Dict[str, RoleAssignment]
    ) -> Dict[str, RoleAssignment]:
        """Stamp victim_model_class into every assignment's metadata.

        All roles (profiler, camouflaguer, sniper, inactive) receive the
        detected victim model class so that each worker can tailor its
        prompt and action logic without needing global coordinator state.
        DIAGNOSTIC assignments are left unchanged (they run before
        classification is complete).
        """

        if self.victim_model_class == VictimModelClass.UNKNOWN:
            return assignments

        annotated: Dict[str, RoleAssignment] = {}
        for aid, a in assignments.items():
            if a.role == AgentRole.DIAGNOSTIC:
                # Probe assignments are issued before classification — skip.
                annotated[aid] = a
                continue
            meta = dict(a.metadata)
            victim_class = self.victim_model_class
            if (
                a.role == AgentRole.SNIPER
                and self.runtime_config.transfer_sniper_direct_target
                and self.victim_model_class == VictimModelClass.LIGHTGCN_STYLE
            ):
                # Override LightGCN-style sniper to MF-style direct target when
                # transfer needs target-positive injections.
                victim_class = VictimModelClass.MF_STYLE
            meta["victim_model_class"] = victim_class.value
            annotated[aid] = RoleAssignment(
                step=a.step,
                agent_id=a.agent_id,
                role=a.role,
                rationale=a.rationale,
                metadata=meta,
            )
        return annotated

    def assign_roles(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        """Delegate role assignment to the configured policy.

        During the probe phase (first ``_probe_steps`` steps) this method
        returns diagnostic assignments and classifies the victim model.  After
        classification it delegates to the configured policy and annotates
        sniper assignments with the detected victim model class so workers can
        choose the correct sniper variant.

        Args:
            observation: Environment state seen by the coordinator.
            worker_states: Current per-agent mutable worker states.

        Returns:
            A mapping from agent ID to role assignment for the step.
        """

        # Probe phase: may return early with diagnostic assignments.
        if not self._probe_phase_done:
            probe_result = self._probe_assign(observation, worker_states)
            if probe_result is not None:
                self._last_assignments = probe_result
                self.last_runtime_trace = {
                    "sniper_lockouts": {},
                    "profiler_forced": False,
                    "probe_phase": True,
                    "victim_model_class": self.victim_model_class.value,
                    "probe_results": list(self._probe_results),
                }
                return probe_result

        # Exploit phase: let the policy decide, then apply post-processing.
        # Keep the LLM policy in sync with the detected model class.
        if hasattr(self.policy, "victim_model_class"):
            self.policy.victim_model_class = self.victim_model_class.value

        assignments = self.policy.assign(observation, worker_states)
        self._update_sniper_lockouts(observation)
        assignments = self._apply_lockouts(observation, assignments)
        assignments = self._ensure_profiler_presence(observation, worker_states, assignments)
        assignments = self._annotate_victim_class(assignments)
        self._last_assignments = assignments
        self.last_runtime_trace = {
            "sniper_lockouts": dict(self._sniper_lockouts),
            "profiler_forced": any(a.role == AgentRole.PROFILER for a in assignments.values()),
            "probe_phase": False,
            "victim_model_class": self.victim_model_class.value,
            "probe_results": list(self._probe_results),
        }
        return assignments
