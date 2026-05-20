"""Coordinator policies for AGAS role assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence

from agas.agents.messages import AgentRole, CoordinatorObservation, RoleAssignment, VictimModelClass
from agas.agents.worker import WorkerState
from agas.llm.prompt_store import PromptStore
from agas.llm.providers import LLMClient, LLMRequest, LLMResponse
from agas.strategies import Strategy

# ---------------------------------------------------------------------------
# Named strategies — each meaningful coordinator behaviour is given a stable
# identifier so it can be counted in ablation tables and toggled via
# ``--strategic-disable-<name>`` flags.
# ---------------------------------------------------------------------------

STRATEGY_PROBE_CLASSIFY = "probe_classify"

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
        "- Snipers rate pre-computed bridge items at 5.0 to create 2-hop paths to the target.\n"
        "- If 'precomputed_bridge_items' appears in the context, bridge items are ALREADY SELECTED "
        "before the episode — skip profiler warm-up entirely and assign snipers from step 0.\n"
        "- NEVER wait for rank improvement before deploying snipers — graph diffusion is gradual "
        "and rank will NOT move during profiler warm-up.\n"
        "- When bridge items are pre-computed: assign snipers from step 0, no profiler rounds needed.\n"
        "- When bridge items are NOT pre-computed: steps 0-2 profiler; steps 3+ sniper+camouflageur.\n"
        "- Keep total sniper interactions per agent low — graph diffusion accumulates slowly."
    ),
    VictimModelClass.SEQUENTIAL_STYLE.value: (
        "- Profiler and camouflageur steps are critical: they build the fake user's interaction "
        "history that makes the eventual target rating feel natural.\n"
        "- Sniper must rate the target LAST in each step (after filler items).\n"
        "- Maintain a steady profiler \u2192 camouflageur cadence to extend history length.\n"
        "- Assign sniper only after at least 2 prior steps of history-building per agent."
    ),
}



# ---------------------------------------------------------------------------
# Named strategies — each meaningful coordinator behaviour is given a stable
# identifier so it can be counted in ablation tables and toggled via
# ``--strategic-disable-<name>`` flags.
# ---------------------------------------------------------------------------

STRATEGY_PROBE_CLASSIFY = "probe_classify"

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
        "- Snipers should rate cluster-neighbour/bridge items at 5.0 and competitors at 5.0.\n"
        "- IMPORTANT: do NOT wait for rank improvement before deploying snipers — graph "
        "diffusion is gradual and rank will NOT move during profiler warm-up. Escalate to "
        "snipers after step 2, even if rank has not improved yet.\n"
        "- Typical progression: steps 0-2 → profiler builds graph proximity; "
        "steps 3+ → mix of sniper (cluster neighbours) and camouflageur; "
        "do NOT stay in warm-up for more than 2-3 consecutive steps.\n"
        "- Budget awareness: with a SMALL agent budget (few fake users), profiler and "
        "camouflaguer must first build graph proximity before snipers fire. With a LARGE "
        "budget (many real users), lighter profiling is sufficient.\n"
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


def _infer_strategy(
    observation: CoordinatorObservation,
    assignments: Dict[str, RoleAssignment],
    step: int,
    probe_phase_done: bool,
) -> str:
    """Infer which paper strategy best describes the current role assignment.

    Maps the LLM coordinator's role-assignment pattern and observation signals
    to one of the eight paper-aligned Strategy labels (S1–S8).

    Args:
        observation: Current environment observation (signals, alerts, rank).
        assignments: Role assignments just produced by the coordinator policy.
        step: Current episode step index.
        probe_phase_done: Whether the victim-model probe phase has completed.

    Returns:
        Strategy enum value string (e.g. ``"S8_MAIN_ATTACK"``).
    """
    roles = [a.role for a in assignments.values()]
    n_sniper = roles.count(AgentRole.SNIPER)
    n_profiler = roles.count(AgentRole.PROFILER)
    n_camouflageur = roles.count(AgentRole.CAMOUFLAGEUR)
    has_alert = bool(observation.alerts_by_agent)
    # Probe steps are PROFILER assignments carrying diagnostic_type in metadata.
    has_probe = any(a.metadata.get("diagnostic_type") for a in assignments.values())

    if has_probe or not probe_phase_done:
        return Strategy.S1_VICTIM_PROBE.value

    max_suspicion = max(
        (
            float((observation.signals_by_agent.get(aid) or {}).get("suspected_filtering_score", 0.0))
            for aid in assignments
        ),
        default=0.0,
    )
    max_discount = max(
        (
            float((observation.signals_by_agent.get(aid) or {}).get("discount_rate", 0.0))
            for aid in assignments
        ),
        default=0.0,
    )

    if has_alert:
        return Strategy.S7_SAFE_REPLACEMENT.value
    if max_suspicion >= 0.5:
        return Strategy.S6_PROFILE_CLEANUP.value
    if n_sniper == 0:
        if max_discount > 0.1:
            return Strategy.S5_SILENT_SLOWDOWN.value
        return Strategy.S3_WARM_UP.value
    if step <= 3:
        return Strategy.S4_FIRST_PUSH.value
    return Strategy.S8_MAIN_ATTACK.value


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
    # SUSPICION_LOCKOUT overlay toggle. When False, ``Coordinator._apply_lockouts``
    # is bypassed (the multi-step lockout decay still runs so the trace stays
    # consistent, but no agent is forced off SNIPER).
    enable_suspicion_lockout: bool = True



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
    # Pre-computed bridge items to inject into the coordinator context.
    # When set, the coordinator is told items are already found and should skip warm-up.
    precomputed_bridge_items: list | None = None

    def __post_init__(self) -> None:
        """Initialize coordinator prompt store and trace cache."""

        self.prompt_store = self.prompt_store or PromptStore()
        self.last_trace: Dict[str, Any] | None = None
        self._unified_memory = None  # type: ignore[assignment]
        self._token_totals: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # Set by Coordinator after probe phase completes so strategy inference is accurate.
        self._probe_phase_done: bool = False

    def set_unified_memory(self, memory) -> None:
        """Attach shared ``UnifiedMemory`` so the coordinator reads/writes it."""

        self._unified_memory = memory

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
        step = observation.step

        def _all_inactive() -> Dict[str, RoleAssignment]:
            return {
                aid: RoleAssignment(
                    step=step,
                    agent_id=aid,
                    role=AgentRole.INACTIVE,
                    rationale="LLM coordinator unavailable; holding all agents inactive.",
                )
                for aid in self.agent_order
            }

        try:
            llm_response = self.client.generate(
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
                "token_usage": None,
            }
            assignments = _all_inactive()
            if self._unified_memory is not None:
                self._unified_memory.append(
                    {
                        "step": int(observation.step),
                        "actor": "coordinator",
                        "kind": "role_assignment",
                        "assignments": {
                            aid: assignments[aid].role.value for aid in self.agent_order
                        },
                        "target_rank": int(observation.target_rank),
                        "fallback": "llm_error",
                    }
                )
            return assignments
        response = llm_response.text
        usage = llm_response.usage
        self._token_totals["prompt_tokens"] += usage.prompt_tokens
        self._token_totals["completion_tokens"] += usage.completion_tokens
        self._token_totals["total_tokens"] += usage.total_tokens
        self.last_trace = {
            "prompt_key": bundle.key,
            "system_prompt": bundle.system_prompt,
            "user_prompt": user_prompt,
            "raw_response": response,
            "fallback_used": False,
            "system_path": bundle.system_path,
            "user_path": bundle.user_path,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

        assignments = self._parse_assignments(response, observation.step)
        if not assignments:
            self.last_trace["fallback_used"] = True
            self.last_trace["fallback_reason"] = "parse_failed"
            assignments = _all_inactive()
        self.last_trace["strategy"] = _infer_strategy(
            observation, assignments, observation.step, self._probe_phase_done
        )
        if self._unified_memory is not None:
            self._unified_memory.append(
                {
                    "step": int(observation.step),
                    "actor": "coordinator",
                    "kind": "role_assignment",
                    "assignments": {
                        aid: assignments[aid].role.value for aid in self.agent_order
                    },
                    "target_rank": int(observation.target_rank),
                }
            )
        return assignments

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
        if self.precomputed_bridge_items:
            state_blob["precomputed_bridge_items"] = list(self.precomputed_bridge_items[:30])
            state_blob["bridge_items_note"] = (
                "Bridge items have been pre-selected from the interaction graph before the episode. "
                "NO warm-up profiling rounds are needed — assign snipers immediately to rate these items."
            )
        if self._unified_memory is not None:
            state_blob["unified_memory"] = self._unified_memory.snapshot()
            state_blob["memory_mode"] = "unified"
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

        # Accept both flat {"agent_1": "sniper", ...} and nested
        # {"strategy": "...", "assignments": {"agent_1": "sniper", ...}} formats.
        if isinstance(data.get("assignments"), dict):
            data = data["assignments"]

        result: Dict[str, RoleAssignment] = {}
        for aid in self.agent_order:
            entry = data.get(aid, "inactive")
            if isinstance(entry, dict):
                role_str = str(entry.get("role", "inactive")).lower().strip()
                metadata = {k: v for k, v in entry.items() if k != "role"}
            else:
                role_str = str(entry).lower().strip()
                metadata = {}
            # Normalize LLM abbreviations: PR→profiler, CA→camouflaguer, SN→sniper, IN→inactive.
            _role_aliases = {
                "pr": "profiler",
                "ca": "camouflaguer", "camouflageur": "camouflaguer",
                "sn": "sniper",
                "in": "inactive",
                "diagnostic": "inactive",
            }
            role_str = _role_aliases.get(role_str, role_str)
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

        # COOCCURRENCE_BRIDGING overlay state (item-axis). When non-empty,
        # the value is stamped into every non-diagnostic assignment's
        # metadata so workers/env can pick the corresponding bridge-item
        # selection method ("cooccurrence" / "gradient" / "auto").
        self._cooccurrence_bridging_method: str | None = None
        self._cooccurrence_bridging_enabled: bool = True

        # Auto-attach so policies can call back for PROBE_CLASSIFY
        # without requiring separate ``attach_coordinator`` plumbing.
        if hasattr(self.policy, "attach_coordinator"):
            try:
                self.policy.attach_coordinator(self)
            except Exception:
                pass

    def configure_cooccurrence_bridging(
        self, method: str | None, enabled: bool = True
    ) -> None:
        """Configure the COOCCURRENCE_BRIDGING overlay.

        Args:
            method: ``"cooccurrence"``, ``"gradient"``, ``"auto"``, or
                ``None``/``"none"`` to disable the metadata stamp.
            enabled: When False the overlay is bypassed even if a method
                is set. Used by ``--strategic-disable-cooccurrence``.
        """

        normalized = None
        if method is not None:
            text = str(method).strip().lower()
            if text and text != "none":
                normalized = text
        self._cooccurrence_bridging_method = normalized
        self._cooccurrence_bridging_enabled = bool(enabled)

    def _apply_cooccurrence_bridging(
        self, assignments: Dict[str, RoleAssignment]
    ) -> Dict[str, RoleAssignment]:
        """Stamp the bridge-item selection method onto each assignment's metadata.

        This is an item-axis overlay that commutes with the role-axis
        overlays. Workers/env continue to read the metadata as today; the
        overlay simply makes the choice traceable in the strategy ledger.
        """

        if (
            not self._cooccurrence_bridging_enabled
            or not self._cooccurrence_bridging_method
        ):
            return assignments
        method = self._cooccurrence_bridging_method
        for aid, a in assignments.items():
            if a.metadata.get("diagnostic_type"):
                continue  # don't stamp bridge method onto probe assignments
            if a.metadata.get("bridge_method") == method:
                continue
            a.metadata["bridge_method"] = method
        return assignments

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

    def _apply_validator_guardrail(
        self,
        observation: CoordinatorObservation,
        assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Force INACTIVE on agents flagged by ProfileValidator.

        The episode runner injects per-agent scores into ``signals_by_agent[aid]``
        under ``validator_flagged``; this wrapper enforces the cool-down
        consistently at the ``Coordinator`` layer.
        """

        for aid, assignment in list(assignments.items()):
            signal = observation.signals_by_agent.get(aid, {})
            if isinstance(signal, dict) and signal.get("validator_flagged", False):
                assignments[aid] = RoleAssignment(
                    step=assignment.step,
                    agent_id=aid,
                    role=AgentRole.INACTIVE,
                    rationale=(
                        "Profile validator flagged this agent "
                        f"(aggregate={signal.get('validator', {}).get('aggregate', 'n/a')}); "
                        "cool down to reduce detectability."
                    ),
                )
        return assignments

    def _apply_lockouts(
        self,
        observation: CoordinatorObservation,
        assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Force locked agents away from sniper roles."""

        if not self.runtime_config.enable_suspicion_lockout:
            return assignments
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

    def probe_classify(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment] | None:
        """Public entry point invoked by policies at PROBE_CLASSIFY priority 1.

        Returns probe-phase diagnostic assignments while probing is in
        progress, or ``None`` once classification is complete. The internal
        state machine still lives on the coordinator because rank-before
        markers must persist across steps.

        Idempotent within a single step — repeated calls return the same
        cached result so policy + coordinator can both invoke it without
        double-advancing the state machine.
        """

        cached_step = getattr(self, "_probe_invoked_step", -1)
        if cached_step == int(observation.step):
            return getattr(self, "_probe_invoked_result", None)
        self._probe_invoked_step = int(observation.step)
        result = self._probe_classify(observation, worker_states)
        self._probe_invoked_result = result
        return result

    def _probe_classify(
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
                role=AgentRole.PROFILER,
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
                        role=AgentRole.PROFILER,
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
                    role=AgentRole.PROFILER,
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
        Probe assignments (profiler with diagnostic_type metadata) are left
        unchanged since they run before classification is complete.
        """

        if self.victim_model_class == VictimModelClass.UNKNOWN:
            return assignments

        annotated: Dict[str, RoleAssignment] = {}
        for aid, a in assignments.items():
            if a.metadata.get("diagnostic_type"):
                # Probe assignments are issued before classification — skip victim-class stamp.
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

        The policy is the single dispatch point: when probing is still in
        progress, the policy invokes ``coordinator.probe_classify`` itself
        as PROBE_CLASSIFY (priority 1).  Once the probe phase completes,
        the same call falls through to the policy's normal phase
        dispatcher and the coordinator-side overlays
        (``SUSPICION_LOCKOUT``, ``VALIDATOR_GUARDRAIL``,
        ``COOCCURRENCE_BRIDGING``) are applied on top.

        Args:
            observation: Environment state seen by the coordinator.
            worker_states: Current per-agent mutable worker states.

        Returns:
            A mapping from agent ID to role assignment for the step.
        """

        # Keep the policy in sync with the latest detected model class
        # and probe phase state before it dispatches.
        if hasattr(self.policy, "victim_model_class"):
            self.policy.victim_model_class = self.victim_model_class.value
        if hasattr(self.policy, "_probe_phase_done"):
            self.policy._probe_phase_done = self._probe_phase_done

        # Priority 1 — PROBE_CLASSIFY. Coordinator drives the probe directly
        # so the state machine advances before the LLM policy dispatches.
        # The ``probe_classify`` call is idempotent within a step.
        if not self._probe_phase_done:
            probe_result = self.probe_classify(observation, worker_states)
            if probe_result is not None:
                if hasattr(self.policy, "last_strategy"):
                    self.policy.last_strategy = STRATEGY_PROBE_CLASSIFY
                if hasattr(self.policy, "last_trace"):
                    self.policy.last_trace = {
                        "strategy": STRATEGY_PROBE_CLASSIFY,
                        "step": int(observation.step),
                    }
                self._last_assignments = probe_result
                self.last_runtime_trace = {
                    "sniper_lockouts": {},
                    "profiler_forced": False,
                    "probe_phase": True,
                    "victim_model_class": self.victim_model_class.value,
                    "probe_results": list(self._probe_results),
                }
                return probe_result

        assignments = self.policy.assign(observation, worker_states)

        # Exploit phase overlays.
        self._update_sniper_lockouts(observation)
        assignments = self._apply_lockouts(observation, assignments)  # SUSPICION_LOCKOUT
        assignments = self._apply_validator_guardrail(observation, assignments)  # VALIDATOR_GUARDRAIL
        assignments = self._ensure_profiler_presence(observation, worker_states, assignments)
        assignments = self._apply_cooccurrence_bridging(assignments)  # COOCCURRENCE_BRIDGING
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
