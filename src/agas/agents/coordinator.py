"""Coordinator policies for AGAS role assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence

from agas.agents.messages import AgentRole, CoordinatorObservation, RoleAssignment, VictimModelClass
from agas.agents.worker import WorkerState
from agas.llm.prompt_store import PromptStore
from agas.llm.providers import LLMClient, LLMRequest

# ---------------------------------------------------------------------------
# Named strategies — each meaningful coordinator behaviour is given a stable
# identifier so it can be counted in ablation tables and toggled via
# ``--strategic-disable-<name>`` flags.
# ---------------------------------------------------------------------------

# Phase strategies (mutually exclusive — one fires per step, priority order).
STRATEGY_PROBE_CLASSIFY = "probe_classify"
STRATEGY_TRUST_BANK_OPENING = "trust_bank_opening"
STRATEGY_SYNCHRONIZED_PAYLOAD = "synchronized_payload"
STRATEGY_BUDGET_PRESSURE = "budget_pressure"
STRATEGY_STEALTH_REBUILD = "stealth_rebuild"
STRATEGY_CONSENSUS_HOLD = "consensus_hold"
STRATEGY_ALERT_COOLDOWN = "alert_cooldown"
STRATEGY_TRUST_RANK_EXPLOIT = "trust_rank_exploit"

# Overlay strategies (post-process; can stack on top of the chosen phase).
STRATEGY_CO_VOTING_DIVERSITY = "co_voting_diversity"
STRATEGY_VALIDATOR_GUARDRAIL = "validator_guardrail"
STRATEGY_SUSPICION_LOCKOUT = "suspicion_lockout"
STRATEGY_COOCCURRENCE_BRIDGING = "cooccurrence_bridging"

PHASE_STRATEGIES = (
    STRATEGY_PROBE_CLASSIFY,
    STRATEGY_TRUST_BANK_OPENING,
    STRATEGY_SYNCHRONIZED_PAYLOAD,
    STRATEGY_BUDGET_PRESSURE,
    STRATEGY_STEALTH_REBUILD,
    STRATEGY_CONSENSUS_HOLD,
    STRATEGY_ALERT_COOLDOWN,
    STRATEGY_TRUST_RANK_EXPLOIT,
)

OVERLAY_STRATEGIES = (
    STRATEGY_CO_VOTING_DIVERSITY,
    STRATEGY_VALIDATOR_GUARDRAIL,
    STRATEGY_SUSPICION_LOCKOUT,
    STRATEGY_COOCCURRENCE_BRIDGING,
)

ALL_STRATEGIES = PHASE_STRATEGIES + OVERLAY_STRATEGIES

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
    # SUSPICION_LOCKOUT overlay toggle. When False, ``Coordinator._apply_lockouts``
    # is bypassed (the multi-step lockout decay still runs so the trace stays
    # consistent, but no agent is forced off SNIPER).
    enable_suspicion_lockout: bool = True


@dataclass
class RuleBasedCoordinatorPolicy:
    """Rule-based policy that reproduces the described AGAS episode logic.

    Every step now executes exactly one *phase strategy* (one of
    ``TRUST_BANK_OPENING``, ``SYNCHRONIZED_PAYLOAD``, ``ALERT_COOLDOWN``,
    ``TRUST_RANK_EXPLOIT``) followed by the ``VALIDATOR_GUARDRAIL`` overlay
    so each branch can be counted in the ablation table.  The dispatch
    order, triggers, and produced assignments are byte-identical to the
    pre-refactor logic when no overlay flags are toggled.
    """

    agent_order: Sequence[str] | None = None
    max_snipers: int = 1
    # Warmup period: keep all agents in profiler/camouflageur for this many steps
    # before allowing any snipers.  Useful for dense-profiler mode where fake users
    # need to accumulate graph-neighbourhood edges first.  Default 0 = no warmup
    # (snipers allowed from step 2 as before).
    sniper_start_step: int = 0
    # Per-strategy enable flags — flip to False for ablation runs.
    enable_trust_bank_opening: bool = True
    enable_synchronized_payload: bool = True
    enable_alert_cooldown: bool = True
    enable_trust_rank_exploit: bool = True
    enable_validator_guardrail: bool = True

    def __post_init__(self) -> None:
        # Trace state populated during ``assign`` so callers can read which
        # named strategy fired this step.
        self.last_strategy: str = "default"
        self.last_trace: Dict[str, Any] = {"strategy": "default"}
        # Set by ``Coordinator.attach_policy`` so the policy can invoke
        # ``coordinator.probe_classify`` at PROBE_CLASSIFY priority 1.
        self._coordinator_ref: Any = None
        self.enable_probe_classify: bool = True

    def attach_coordinator(self, coordinator: Any) -> None:
        """Allow the policy to call back into ``coordinator.probe_classify``."""

        self._coordinator_ref = coordinator

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
    def _validator_flagged(observation: CoordinatorObservation, agent_id: str) -> bool:
        """Return whether the profile validator flagged this agent this step."""

        signal = observation.signals_by_agent.get(agent_id, {})
        if not isinstance(signal, dict):
            return False
        return bool(signal.get("validator_flagged", False))

    def _inactive_baseline(
        self,
        observation: CoordinatorObservation,
    ) -> Dict[str, RoleAssignment]:
        """All-inactive baseline used as the starting point for every phase."""

        order = self.agent_order or ()
        step = observation.step
        return {
            agent_id: RoleAssignment(
                step=step,
                agent_id=agent_id,
                role=AgentRole.INACTIVE,
                rationale="Default temporal reset to minimize velocity anomaly risk.",
            )
            for agent_id in order
        }

    # ------------------------------------------------------------------
    # Phase strategies
    # ------------------------------------------------------------------

    def _trust_bank_opening(
        self,
        observation: CoordinatorObservation,
        assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Steps 0–1 of the exploit phase: profile-and-camouflage warm-up."""

        order = self.agent_order or ()
        step = observation.step
        if step == 0:
            scout = order[0]
            assignments[scout] = RoleAssignment(
                step=step,
                agent_id=scout,
                role=AgentRole.PROFILER,
                rationale="Probe if recommender is accepting new ratings without triggering bans.",
            )
            return assignments
        # step == 1
        assignments[order[0]].role = AgentRole.PROFILER
        assignments[order[0]].rationale = "Continue profiling while trust bank is built."
        for aid in order[1:3]:
            assignments[aid].role = AgentRole.CAMOUFLAGEUR
            assignments[aid].rationale = "Build target-domain trust embedding before payload delivery."
        return assignments

    def _synchronized_payload(
        self,
        observation: CoordinatorObservation,
        assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Step 2: first synchronized sniper volley + camouflage tail."""

        order = self.agent_order or ()
        # Prefer keeping agent 0 as a background camouflaguer, but guarantee at
        # least one sniper even when running with 1–2 agents.
        sniper_pool = list(order[1:]) or [order[0]]
        for aid in sniper_pool[:2]:
            assignments[aid].role = AgentRole.SNIPER
            assignments[aid].rationale = "Deliver synchronized payload after camouflage warm-up."
        remaining = [aid for aid in order if assignments[aid].role == AgentRole.INACTIVE]
        for aid in remaining[:1]:
            assignments[aid].role = AgentRole.CAMOUFLAGEUR
            assignments[aid].rationale = "Generate benign background activity as timing noise."
        return assignments

    def _alert_cooldown(
        self,
        observation: CoordinatorObservation,
        alerted: Sequence[str],
        assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Reactive evasion when an alert or high suspicion is already visible."""

        order = self.agent_order or ()
        first = alerted[0]
        assignments[first].role = AgentRole.CAMOUFLAGEUR
        assignments[first].rationale = "Cool down suspicious profile with normal domain behavior."
        if len(alerted) > 1:
            second = alerted[1]
            assignments[second].role = AgentRole.INACTIVE
            assignments[second].rationale = "Temporary dormancy to drop velocity/risk scores."
        for aid in order:
            if aid not in alerted and assignments[aid].role == AgentRole.INACTIVE:
                assignments[aid].role = AgentRole.CAMOUFLAGEUR
                assignments[aid].rationale = "Rotate fresh camouflaguer to continue trust accumulation."
                break
        return assignments

    def _trust_rank_exploit(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
        assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Default exploit loop: top-(trust−risk) → SNIPER, rest → CAMOUFLAGEUR."""

        order = self.agent_order or ()
        ranked = sorted(
            order,
            key=lambda aid: worker_states[aid].trust - worker_states[aid].risk,
            reverse=True,
        )
        step = observation.step
        if observation.target_rank > 5 and ranked and step >= self.sniper_start_step:
            max_snipers = max(0, int(self.max_snipers))
            for best in ranked[:max_snipers]:
                assignments[best].role = AgentRole.SNIPER
                assignments[best].rationale = "Use the strongest trusted profile for the next payload step."

        support_pool = [aid for aid in ranked if assignments[aid].role == AgentRole.INACTIVE]
        for aid in support_pool[:3]:
            assignments[aid].role = AgentRole.CAMOUFLAGEUR
            assignments[aid].rationale = "Support trust camouflage and cluster alignment."
        return assignments

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _apply_validator_guardrail(
        self,
        observation: CoordinatorObservation,
        assignments: Dict[str, RoleAssignment],
    ) -> tuple[Dict[str, RoleAssignment], int]:
        """Force INACTIVE for any agent flagged by ProfileValidator."""

        if not self.enable_validator_guardrail:
            return assignments, 0
        flagged = 0
        for aid in self.agent_order or ():
            if self._validator_flagged(observation, aid):
                assignments[aid].role = AgentRole.INACTIVE
                assignments[aid].rationale = (
                    "Profile validator flagged this agent; cool down to reduce detectability."
                )
                flagged += 1
        return assignments, flagged

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        """Apply a deterministic role schedule with alert-aware evasive fallback.

        Routes through the named phase helpers in the same priority order
        as the original implementation:

            TRUST_BANK_OPENING → SYNCHRONIZED_PAYLOAD → ALERT_COOLDOWN
                              → TRUST_RANK_EXPLOIT

        followed by the ``VALIDATOR_GUARDRAIL`` overlay.

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

        # Priority 1 — PROBE_CLASSIFY. Delegate to the coordinator's probe
        # state machine when probing is still in progress so the strategy
        # trace records the probe phase under a named row instead of being
        # silent.
        if (
            self.enable_probe_classify
            and self._coordinator_ref is not None
            and not getattr(self._coordinator_ref, "_probe_phase_done", True)
        ):
            probe_result = self._coordinator_ref.probe_classify(observation, worker_states)
            if probe_result is not None:
                self.last_strategy = STRATEGY_PROBE_CLASSIFY
                self.last_trace = {
                    "strategy": STRATEGY_PROBE_CLASSIFY,
                    "step": int(observation.step),
                }
                return probe_result

        assignments = self._inactive_baseline(observation)
        step = observation.step

        # TRUST_BANK_OPENING (steps 0–1)
        if self.enable_trust_bank_opening and step <= 1:
            self.last_strategy = STRATEGY_TRUST_BANK_OPENING
            self.last_trace = {"strategy": STRATEGY_TRUST_BANK_OPENING, "step": int(step)}
            return self._trust_bank_opening(observation, assignments)

        # SYNCHRONIZED_PAYLOAD (step 2, post-warmup)
        if (
            self.enable_synchronized_payload
            and step == 2
            and step >= self.sniper_start_step
        ):
            self.last_strategy = STRATEGY_SYNCHRONIZED_PAYLOAD
            self.last_trace = {"strategy": STRATEGY_SYNCHRONIZED_PAYLOAD, "step": int(step)}
            return self._synchronized_payload(observation, assignments)

        # ALERT_COOLDOWN (any step with fresh alerts or high suspicion)
        alerted = [
            aid
            for aid in (self.agent_order or ())
            if aid in observation.alerts_by_agent or self._black_box_suspicion(observation, aid) >= 0.55
        ]
        if self.enable_alert_cooldown and alerted:
            self.last_strategy = STRATEGY_ALERT_COOLDOWN
            self.last_trace = {
                "strategy": STRATEGY_ALERT_COOLDOWN,
                "alerted": list(alerted),
            }
            return self._alert_cooldown(observation, alerted, assignments)

        # TRUST_RANK_EXPLOIT (default loop) + VALIDATOR_GUARDRAIL overlay.
        if self.enable_trust_rank_exploit:
            self.last_strategy = STRATEGY_TRUST_RANK_EXPLOIT
            self.last_trace = {"strategy": STRATEGY_TRUST_RANK_EXPLOIT}
            assignments = self._trust_rank_exploit(observation, worker_states, assignments)
        else:
            self.last_strategy = "default"
            self.last_trace = {"strategy": "default"}

        assignments, flagged_count = self._apply_validator_guardrail(observation, assignments)
        if flagged_count > 0:
            self.last_trace.setdefault("validator_guardrail_flags", flagged_count)

        return assignments


@dataclass
class StrategicCoordinatorPolicy:
    """Composes ``RuleBasedCoordinatorPolicy`` with the adaptive strategies.

    Phase priority (first match wins):
        1. PROBE_CLASSIFY     — coordinator-driven victim-model probe
        2. TRUST_BANK_OPENING — exploit-phase warm-up (steps 0–1)
        3. SYNCHRONIZED_PAYLOAD — first synchronized sniper volley
        4. BUDGET_PRESSURE    — silent acceptance erosion
        5. STEALTH_REBUILD    — proactive cool-down
        6. CONSENSUS_HOLD     — mid-episode re-probe
        7. ALERT_COOLDOWN     — reactive cool-down (also arms SUSPICION_LOCKOUT)
        8. TRUST_RANK_EXPLOIT — default trust−risk loop

    Overlays applied after the chosen phase:
        * CO_VOTING_DIVERSITY — rotate over-fired snipers
        * VALIDATOR_GUARDRAIL — per-agent INACTIVE veto (in fallback)
        * SUSPICION_LOCKOUT   — multi-step lockout (Coordinator-side)
        * COOCCURRENCE_BRIDGING — LightGCN item-selection metadata stamp
    """

    agent_order: Sequence[str] | None = None
    max_snipers: int = 1
    sniper_start_step: int = 0
    # Strategy 1 thresholds (post-tuning defaults: stricter trigger,
    # rate-limited firing, partial suppression).
    stealth_aggregate_mean: float = 0.70
    stealth_streak: int = 2
    stealth_collusion_lock_steps: int = 2
    stealth_cooldown_steps: int = 3
    enable_stealth_rebuild: bool = True
    # Strategy 2 thresholds.
    reprobe_window: int = 3
    reprobe_min_step: int = 4
    enable_consensus_hold: bool = True
    # Strategy 3 (BUDGET_PRESSURE).
    budget_pressure_min_step: int = 4
    budget_pressure_drop_ratio: float = 0.7  # current must be <70% of early baseline
    budget_pressure_hold_steps: int = 2
    enable_budget_pressure: bool = True
    # Strategy 4 (CO_VOTING_DIVERSITY): agent-rotation form.
    diversity_lookback: int = 3
    diversity_max_repeat: int = 2  # demote any sniper assigned in >= 2 of last lookback rounds
    enable_diversity: bool = True
    # Phase strategies inherited from the rule-based fallback.
    enable_probe_classify: bool = True
    enable_trust_bank_opening: bool = True
    enable_synchronized_payload: bool = True
    enable_alert_cooldown: bool = True
    enable_trust_rank_exploit: bool = True
    # Overlays.
    enable_validator_guardrail: bool = True
    enable_suspicion_lockout: bool = True
    enable_cooccurrence_bridging: bool = True
    # Default item-selection method for the COOCCURRENCE_BRIDGING overlay.
    # ``None`` leaves whatever the environment-side default is in place.
    bridge_method_default: str | None = None

    def __post_init__(self) -> None:
        self._fallback = RuleBasedCoordinatorPolicy(
            agent_order=self.agent_order,
            max_snipers=self.max_snipers,
            sniper_start_step=self.sniper_start_step,
            enable_trust_bank_opening=self.enable_trust_bank_opening,
            enable_synchronized_payload=self.enable_synchronized_payload,
            enable_alert_cooldown=self.enable_alert_cooldown,
            enable_trust_rank_exploit=self.enable_trust_rank_exploit,
            enable_validator_guardrail=self.enable_validator_guardrail,
        )
        self._reprobe_used: bool = False
        self._reprobe_step: int | None = None
        self._reprobe_rank_before: int | None = None
        self._rank_history: list[int] = []
        self._delta_history: list[int] = []
        self._stealth_cooldown_remaining: int = 0
        # Strategy 3/4 internal state.
        self._accept_history: list[tuple[int, int]] = []  # (accepted, attempted) per step
        self._budget_pressure_remaining: int = 0
        self._sniper_role_history: list[set[str]] = []  # set of sniper agent IDs per step
        # Filled by Coordinator after assignment so paper-grade traces survive.
        self.last_strategy: str = "default"
        self.last_trace: Dict[str, Any] = {}
        self.victim_model_class: str = VictimModelClass.UNKNOWN.value
        # Provided by Coordinator each step so CONSENSUS_HOLD can flip it
        # and PROBE_CLASSIFY can be invoked at priority 1.
        self._coordinator_ref: Any = None

    def attach_coordinator(self, coordinator: Any) -> None:
        """Coordinator passes self so we can flip ``victim_model_class``."""

        self._coordinator_ref = coordinator

    def _validator_stats(
        self, observation: CoordinatorObservation
    ) -> tuple[float, float, str | None]:
        """Return (mean_aggregate, max_streak, top_collusion_agent)."""

        agg_vals: list[float] = []
        streaks: list[int] = []
        top_collusion = (-1.0, None)
        for aid, signal in observation.signals_by_agent.items():
            if not isinstance(signal, dict):
                continue
            v = signal.get("validator", {})
            if isinstance(v, dict):
                agg_vals.append(float(v.get("aggregate", 0.0)))
                col = float(v.get("collusion", 0.0))
                if col > top_collusion[0]:
                    top_collusion = (col, aid)
            streaks.append(int(signal.get("suppression_streak", 0)))
        mean_agg = sum(agg_vals) / max(1, len(agg_vals))
        max_streak = max(streaks) if streaks else 0
        return mean_agg, float(max_streak), top_collusion[1]

    def _needs_stealth_rebuild(self, observation: CoordinatorObservation) -> bool:
        if not self.enable_stealth_rebuild:
            return False
        if self._stealth_cooldown_remaining > 0:
            return False
        mean_agg, max_streak, _ = self._validator_stats(observation)
        if mean_agg >= self.stealth_aggregate_mean:
            return True
        if max_streak >= self.stealth_streak:
            return True
        if observation.alerts_by_agent:
            return True
        return False

    def _stealth_rebuild(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
        baseline_assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Partial suppression: lock the *highest-aggregate* sniper only.

        Other roles assigned by the underlying rule-based policy are
        preserved, so the attack budget is not starved. The cleanest
        profile is also nudged to ``CAMOUFLAGEUR`` if it is currently
        ``INACTIVE`` so total throughput stays roughly constant.
        """

        step = observation.step
        order = self.agent_order or tuple(sorted(worker_states.keys()))
        _, _, top_collusion_agent = self._validator_stats(observation)

        per_agent_agg: Dict[str, float] = {}
        for aid in order:
            v = observation.signals_by_agent.get(aid, {}).get("validator", {})
            if isinstance(v, dict):
                per_agent_agg[aid] = float(v.get("aggregate", 0.0))
            else:
                per_agent_agg[aid] = 0.0

        # Start from the rule-based plan and mutate selectively (partial
        # suppression rather than blanket INACTIVE).
        assignments: Dict[str, RoleAssignment] = {
            aid: RoleAssignment(
                step=a.step,
                agent_id=a.agent_id,
                role=a.role,
                rationale=a.rationale,
                metadata=dict(a.metadata),
            )
            for aid, a in baseline_assignments.items()
        }

        sniper_ids = [aid for aid, a in assignments.items() if a.role == AgentRole.SNIPER]
        if sniper_ids:
            worst_sniper = max(sniper_ids, key=lambda a: per_agent_agg.get(a, 0.0))
            assignments[worst_sniper] = RoleAssignment(
                step=step,
                agent_id=worst_sniper,
                role=AgentRole.INACTIVE,
                rationale=(
                    "STEALTH_REBUILD: highest-aggregate sniper cooled down "
                    f"(agg={per_agent_agg.get(worst_sniper, 0.0):.2f})."
                ),
            )

        if top_collusion_agent is not None:
            assignments[top_collusion_agent] = RoleAssignment(
                step=step,
                agent_id=top_collusion_agent,
                role=AgentRole.INACTIVE,
                rationale="STEALTH_REBUILD: highest-collusion agent locked.",
            )

        cleanest = min(per_agent_agg, key=lambda a: per_agent_agg[a]) if per_agent_agg else None
        if (
            cleanest is not None
            and cleanest != top_collusion_agent
            and assignments[cleanest].role == AgentRole.INACTIVE
        ):
            assignments[cleanest] = RoleAssignment(
                step=step,
                agent_id=cleanest,
                role=AgentRole.CAMOUFLAGEUR,
                rationale=(
                    "STEALTH_REBUILD: cleanest profile maintains benign activity to avoid full silence."
                ),
            )
        return assignments

    def _record_history(self, observation: CoordinatorObservation) -> None:
        self._rank_history.append(int(observation.target_rank))
        self._delta_history.append(int(observation.target_rank_delta))
        if len(self._rank_history) > 32:
            self._rank_history = self._rank_history[-32:]
            self._delta_history = self._delta_history[-32:]
        accepted = 0
        attempted = 0
        for signal in observation.signals_by_agent.values():
            if isinstance(signal, dict):
                accepted += int(signal.get("accepted_actions", 0))
                attempted += int(signal.get("attempted_actions", 0))
        self._accept_history.append((accepted, attempted))
        if len(self._accept_history) > 32:
            self._accept_history = self._accept_history[-32:]

    def _needs_budget_pressure(self, observation: CoordinatorObservation) -> bool:
        """STRATEGY 3 trigger — silent suppression detection."""

        if not self.enable_budget_pressure:
            return False
        if observation.step < self.budget_pressure_min_step:
            return False
        if observation.alerts_by_agent:
            return False  # already covered by stealth/rule
        for signal in observation.signals_by_agent.values():
            if isinstance(signal, dict) and int(signal.get("suppression_streak", 0)) > 0:
                return False  # not silent — streak signal already visible
        if len(self._accept_history) < 4:
            return False
        # Compare recent 3-step acceptance to early 3-step baseline.
        early = self._accept_history[1 : 1 + 3]
        recent = self._accept_history[-3:]
        early_a = sum(a for a, _ in early)
        early_t = sum(t for _, t in early)
        recent_a = sum(a for a, _ in recent)
        recent_t = sum(t for _, t in recent)
        if early_t == 0 or recent_t == 0:
            return False
        early_rate = early_a / early_t
        recent_rate = recent_a / recent_t
        return recent_rate < self.budget_pressure_drop_ratio * early_rate

    def _budget_pressure_assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
        baseline_assignments: Dict[str, RoleAssignment],
    ) -> Dict[str, RoleAssignment]:
        """Force the 2 highest-trust / lowest-aggregate agents to CAMOUFLAGEUR."""

        step = observation.step
        order = self.agent_order or tuple(sorted(worker_states.keys()))
        per_agent_agg: Dict[str, float] = {}
        for aid in order:
            v = observation.signals_by_agent.get(aid, {}).get("validator", {})
            per_agent_agg[aid] = float(v.get("aggregate", 0.0)) if isinstance(v, dict) else 0.0
        # Score: high trust (positive) and low aggregate (positive). Pick top 2.
        scored = sorted(
            order,
            key=lambda a: (worker_states[a].trust - per_agent_agg.get(a, 0.0)),
            reverse=True,
        )
        promote = set(scored[:2])
        assignments: Dict[str, RoleAssignment] = {
            aid: RoleAssignment(
                step=a.step,
                agent_id=a.agent_id,
                role=a.role,
                rationale=a.rationale,
                metadata=dict(a.metadata),
            )
            for aid, a in baseline_assignments.items()
        }
        for aid in promote:
            assignments[aid] = RoleAssignment(
                step=step,
                agent_id=aid,
                role=AgentRole.CAMOUFLAGEUR,
                rationale=(
                    "BUDGET_PRESSURE: silent acceptance erosion detected; "
                    "cleanest trusted profile rebuilds trust as cover."
                ),
            )
        return assignments

    def _apply_diversity(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
        assignments: Dict[str, RoleAssignment],
    ) -> tuple[Dict[str, RoleAssignment], int]:
        """STRATEGY 4 — agent-rotation diversity.

        Demote any sniper that has been a sniper in >= ``diversity_max_repeat``
        of the last ``diversity_lookback`` rounds; promote the most-rested
        eligible agent in their place. Returns (assignments, swap_count).
        """

        if not self.enable_diversity:
            return assignments, 0
        order = self.agent_order or tuple(sorted(worker_states.keys()))
        recent = self._sniper_role_history[-self.diversity_lookback :]
        rep_count: Dict[str, int] = {aid: 0 for aid in order}
        for snipe_set in recent:
            for aid in snipe_set:
                rep_count[aid] = rep_count.get(aid, 0) + 1
        snipers_now = [aid for aid, a in assignments.items() if a.role == AgentRole.SNIPER]
        if not snipers_now:
            return assignments, 0
        non_sniper = [aid for aid, a in assignments.items() if a.role != AgentRole.SNIPER]
        rested_pool = sorted(non_sniper, key=lambda a: rep_count.get(a, 0))
        swaps = 0
        for sniper_aid in list(snipers_now):
            if rep_count.get(sniper_aid, 0) < self.diversity_max_repeat:
                continue  # not over-fired yet
            # Find most rested eligible swap target.
            replacement = None
            for cand in rested_pool:
                if rep_count.get(cand, 0) < self.diversity_max_repeat:
                    replacement = cand
                    break
            if replacement is None:
                break
            # Swap roles.
            old_role = assignments[replacement].role
            assignments[replacement] = RoleAssignment(
                step=assignments[sniper_aid].step,
                agent_id=replacement,
                role=AgentRole.SNIPER,
                rationale="DIVERSITY: rested agent rotated in to reduce co-voting footprint.",
            )
            assignments[sniper_aid] = RoleAssignment(
                step=assignments[sniper_aid].step,
                agent_id=sniper_aid,
                role=old_role,
                rationale=(
                    f"DIVERSITY: agent fired in {rep_count[sniper_aid]} of last "
                    f"{self.diversity_lookback} rounds; rotating off."
                ),
            )
            rested_pool.remove(replacement)
            swaps += 1
        return assignments, swaps

    def _needs_re_probe(self, observation: CoordinatorObservation) -> bool:
        if not self.enable_consensus_hold or self._reprobe_used:
            return False
        if observation.step < self.reprobe_min_step:
            return False
        if len(self._delta_history) < self.reprobe_window:
            return False
        recent = self._delta_history[-self.reprobe_window :]
        mean_delta = sum(recent) / len(recent)
        vc = self.victim_model_class
        # MF was classified but direct snipers are not improving rank: re-probe.
        if vc == VictimModelClass.MF_STYLE.value and mean_delta <= 0:
            return True
        # LightGCN was classified but bridge strategy is producing strong positive
        # deltas (>0 sustained) — possibly the model behaves more MF-like for this
        # subsample; re-probe to verify.
        if vc == VictimModelClass.LIGHTGCN_STYLE.value and mean_delta > 5:
            return True
        return False

    def _consensus_re_probe(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        """One agent direct-rates target; everyone else inactive."""

        step = observation.step
        order = self.agent_order or tuple(sorted(worker_states.keys()))
        # Pick the agent with lowest validator aggregate (cleanest profile).
        cleanest = min(
            order,
            key=lambda a: float(
                observation.signals_by_agent.get(a, {})
                .get("validator", {})
                .get("aggregate", 0.0)
                if isinstance(observation.signals_by_agent.get(a, {}), dict)
                else 0.0
            ),
        )
        assignments: Dict[str, RoleAssignment] = {}
        for aid in order:
            assignments[aid] = RoleAssignment(
                step=step,
                agent_id=aid,
                role=AgentRole.INACTIVE,
                rationale="CONSENSUS_HOLD: holding agents inactive during mid-episode re-probe.",
            )
        assignments[cleanest] = RoleAssignment(
            step=step,
            agent_id=cleanest,
            role=AgentRole.SNIPER,
            rationale=(
                "CONSENSUS_HOLD: cleanest profile fires direct-target sniper to "
                "test classification under current graph state."
            ),
            metadata={"reprobe": True, "reprobe_type": "direct"},
        )
        self._reprobe_used = True
        self._reprobe_step = step
        self._reprobe_rank_before = int(observation.target_rank)
        return assignments

    def _maybe_finalise_reprobe(self, observation: CoordinatorObservation) -> None:
        """Read result of an earlier re-probe and update victim_model_class."""

        if self._reprobe_step is None or self._reprobe_rank_before is None:
            return
        if observation.step <= self._reprobe_step:
            return
        delta = self._reprobe_rank_before - int(observation.target_rank)
        # delta > 0 means rank improved (lower number == better).
        previous = self.victim_model_class
        if previous == VictimModelClass.MF_STYLE.value and delta <= 0:
            new_class = VictimModelClass.LIGHTGCN_STYLE.value
        elif previous == VictimModelClass.LIGHTGCN_STYLE.value and delta > 0:
            new_class = VictimModelClass.MF_STYLE.value
        else:
            new_class = previous
        if new_class != previous:
            self.victim_model_class = new_class
            if self._coordinator_ref is not None:
                try:
                    self._coordinator_ref.victim_model_class = VictimModelClass(new_class)
                except (ValueError, AttributeError):
                    pass
            self.last_trace.setdefault("reprobe_history", []).append(
                {
                    "step": self._reprobe_step,
                    "delta": delta,
                    "old_class": previous,
                    "new_class": new_class,
                }
            )
        # Reset markers so we don't re-finalise.
        self._reprobe_step = None
        self._reprobe_rank_before = None

    def assign(
        self,
        observation: CoordinatorObservation,
        worker_states: Dict[str, WorkerState],
    ) -> Dict[str, RoleAssignment]:
        if self.agent_order is None:
            self.agent_order = tuple(sorted(worker_states.keys()))
            self._fallback.agent_order = self.agent_order

        # Priority 1 — PROBE_CLASSIFY. The coordinator owns the probe state
        # machine; we delegate to it so the strategy trace records the probe
        # phase and overlays are skipped for diagnostic assignments.
        if (
            self.enable_probe_classify
            and self._coordinator_ref is not None
            and not getattr(self._coordinator_ref, "_probe_phase_done", True)
        ):
            probe_result = self._coordinator_ref.probe_classify(observation, worker_states)
            if probe_result is not None:
                self.last_strategy = STRATEGY_PROBE_CLASSIFY
                self.last_trace = {
                    "strategy": STRATEGY_PROBE_CLASSIFY,
                    "step": int(observation.step),
                    "victim_model_class": getattr(
                        self._coordinator_ref, "victim_model_class", VictimModelClass.UNKNOWN
                    ).value
                    if hasattr(self._coordinator_ref, "victim_model_class")
                    else VictimModelClass.UNKNOWN.value,
                }
                return probe_result

        self._record_history(observation)
        self._maybe_finalise_reprobe(observation)
        if self._stealth_cooldown_remaining > 0:
            self._stealth_cooldown_remaining -= 1
        if self._budget_pressure_remaining > 0:
            self._budget_pressure_remaining -= 1

        # Sync fallback enable flags so ablation toggles flow through.
        self._fallback.enable_trust_bank_opening = self.enable_trust_bank_opening
        self._fallback.enable_synchronized_payload = self.enable_synchronized_payload
        self._fallback.enable_alert_cooldown = self.enable_alert_cooldown
        self._fallback.enable_trust_rank_exploit = self.enable_trust_rank_exploit
        self._fallback.enable_validator_guardrail = self.enable_validator_guardrail

        result: Dict[str, RoleAssignment] | None = None

        # Priority 4 — BUDGET_PRESSURE. Highest priority among adaptives
        # because silent suppression is the earliest detectable signal of
        # soft defenses.
        if self._budget_pressure_remaining > 0 or self._needs_budget_pressure(observation):
            baseline = self._fallback.assign(observation, worker_states)
            if self._budget_pressure_remaining == 0:
                self._budget_pressure_remaining = max(1, int(self.budget_pressure_hold_steps))
            self.last_strategy = STRATEGY_BUDGET_PRESSURE
            self.last_trace = {
                "strategy": STRATEGY_BUDGET_PRESSURE,
                "remaining": int(self._budget_pressure_remaining),
                "accept_history": list(self._accept_history[-3:]),
            }
            result = self._budget_pressure_assign(observation, worker_states, baseline)

        # Priority 5 — STEALTH_REBUILD.
        elif self._needs_stealth_rebuild(observation):
            baseline = self._fallback.assign(observation, worker_states)
            self.last_strategy = STRATEGY_STEALTH_REBUILD
            mean_agg, max_streak, _ = self._validator_stats(observation)
            self.last_trace = {
                "strategy": STRATEGY_STEALTH_REBUILD,
                "mean_aggregate": mean_agg,
                "max_streak": max_streak,
                "alerts": list(observation.alerts_by_agent.keys()),
                "cooldown_set_to": int(self.stealth_cooldown_steps),
            }
            self._stealth_cooldown_remaining = max(0, int(self.stealth_cooldown_steps))
            result = self._stealth_rebuild(observation, worker_states, baseline)

        # Priority 6 — CONSENSUS_HOLD.
        elif self._needs_re_probe(observation):
            self.last_strategy = STRATEGY_CONSENSUS_HOLD
            self.last_trace = {
                "strategy": STRATEGY_CONSENSUS_HOLD,
                "step": int(observation.step),
                "victim_model_class_before": self.victim_model_class,
                "delta_window": list(self._delta_history[-self.reprobe_window :]),
            }
            result = self._consensus_re_probe(observation, worker_states)

        # Priorities 2/3/7/8 — fall through to the rule-based phase
        # dispatcher (TRUST_BANK_OPENING → SYNCHRONIZED_PAYLOAD →
        # ALERT_COOLDOWN → TRUST_RANK_EXPLOIT). The fallback's
        # ``last_strategy`` is bubbled up so the trace counter records the
        # actual phase that fired.
        if result is None:
            result = self._fallback.assign(observation, worker_states)
            self.last_strategy = self._fallback.last_strategy
            self.last_trace = dict(self._fallback.last_trace)

        # Overlay — CO_VOTING_DIVERSITY.
        result, swaps = self._apply_diversity(observation, worker_states, result)
        if swaps > 0:
            self.last_trace.setdefault("diversity_swaps", swaps)

        # Track sniper history for next round's diversity check.
        snipers_now = {aid for aid, a in result.items() if a.role == AgentRole.SNIPER}
        self._sniper_role_history.append(snipers_now)
        if len(self._sniper_role_history) > 32:
            self._sniper_role_history = self._sniper_role_history[-32:]

        return result


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
        self._unified_memory = None  # type: ignore[assignment]

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
            assignments = fallback.assign(observation, worker_states)
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
        if not assignments:
            self.last_trace["fallback_used"] = True
            self.last_trace["fallback_reason"] = "parse_failed"
            assignments = fallback.assign(observation, worker_states)
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
            if a.role == AgentRole.DIAGNOSTIC:
                continue
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

        Works for any coordinator policy (rule-based or LLM): the episode
        runner injects per-agent scores into ``signals_by_agent[aid]``
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
        # before it dispatches.
        if hasattr(self.policy, "victim_model_class"):
            self.policy.victim_model_class = self.victim_model_class.value

        # Priority 1 — PROBE_CLASSIFY. Policies that have a coordinator ref
        # (``RuleBasedCoordinatorPolicy``, ``StrategicCoordinatorPolicy``)
        # invoke ``probe_classify`` themselves at priority 1 inside their
        # ``assign``. For policies that don't (e.g., LLM coordinator,
        # third-party stubs), Coordinator drives the probe directly so the
        # state machine still advances. The ``probe_classify`` call is
        # idempotent within a step, so this is safe to call after policy.
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
