"""Worker agent behaviors for each AGAS role."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Optional, Sequence

from agas.agents.messages import AgentRole, RatingAction, RoleAssignment, WorkerActionReport
from agas.llm.prompt_store import PromptStore
from agas.llm.providers import LLMClient, LLMRequest, LLMResponse


@dataclass
class WorkerState:
    """Mutable worker state tracked across steps."""

    agent_id: str
    trust: float = 0.0
    risk: float = 0.0
    current_role: AgentRole = AgentRole.INACTIVE
    actions_taken: int = 0
    role_history: List[str] = field(default_factory=list)


@dataclass
class WorkerContext:
    """Candidate pools used by worker role policies."""

    target_item_id: str
    benchmark_items: Sequence[str]
    target_cluster_items: Sequence[str]
    competitor_items: Sequence[str]
    noise_items: Sequence[str]
    # Gradient-selected bridge items: items whose propagated LightGCN embedding
    # points most strongly toward the target neighbourhood.  Empty list when
    # gradient selection is disabled or episode model is not LightGCN.
    bridge_items: Sequence[str] = ()


@dataclass
class WorkerPolicyConfig:
    """Counts of actions emitted by each role."""

    profiler_actions: int = 3
    camouflaguer_actions: int = 2
    sniper_competitor_actions: int = 2
    implicit_safe: bool = False
    positive_threshold: float = 4.0
    # Graph-sniper mode: designed for degree-normalized models (e.g. LightGCN).
    # Instead of rating the target directly (which inflates its degree and dilutes
    # all its existing edges), snipers rate cluster-neighbour items at 5.0 so that
    # graph diffusion propagates the signal to the target without touching its degree.
    # Competitors are rated at 5.0 (not 1.0) to inflate their degrees and reduce
    # their edge weights, pushing them lower in the ranking.
    graph_sniper: bool = False
    graph_sniper_neighbor_actions: int = 3
    # For offline transfer attacks, the victim model is retrained from scratch on the
    # augmented dataset — degree normalisation adapts to the new edges, so adding a
    # direct 5.0 target rating at the end of the graph-sniper payload still helps.
    # Set to False only for pure online-injection attacks where the LightGCN model is
    # never retrained and degree normalisation would penalise the direct edge.
    graph_sniper_include_target: bool = True
    # Sequential-sniper mode: for recency-aware models (SASRec, GRU4Rec, BERT4Rec).
    # Rates genre-consistent filler items first to build interaction history, then
    # rates the target item last so the next-item prediction bias fires on the target.
    sequential_sniper: bool = False
    sequential_filler_actions: int = 3
    # LightGCN interaction budget. "small" = few fake users (cold-start, need graph
    # proximity building); "large" = many real users (established graph connections,
    # lighter profiling/camouflage needed before snipers fire).
    lightgcn_budget: str = "small"
    # Dense-profiler mode: build a richer fake-user profile before the sniper fires.
    # Increases graph connectivity for cold-start fake users attacking degree-normalised
    # models (LightGCN/NGCF) without requiring real user IDs or cloned histories.
    # When profiler_use_cluster=True the profiler always draws from target_cluster_items
    # (not benchmark/popular items), regardless of victim_model_class, so the fake user
    # accumulates edges in the target neighbourhood before the sniper action.
    profiler_use_cluster: bool = False


class WorkerAgent:
    """Worker controlled by coordinator role assignments."""

    def __init__(
        self,
        state: WorkerState,
        config: WorkerPolicyConfig | None = None,
        seed: int = 42,
        llm_client: LLMClient | None = None,
        prompt_store: PromptStore | None = None,
        policy_name: str = "openai",
        temperature: float = 0.2,
        temperature_end: float | None = None,
        total_steps: int | None = None,
    ):
        """Initialize one worker with mutable state and deterministic RNG.

        Args:
            state: Mutable state object owned by this worker.
            config: Optional action-count configuration; defaults to
                ``WorkerPolicyConfig``.
            seed: Base random seed used to create deterministic sampling behavior.
            llm_client: Optional LLM backend used for role-specific action generation.
            prompt_store: Optional prompt store for loading per-role templates.
            policy_name: Human-readable policy label for traces and logs.
        """

        self.state = state
        self.config = config or WorkerPolicyConfig()
        self._rand = Random(seed + hash(state.agent_id) % 10_000)
        self._llm_client = llm_client
        self._prompt_store = prompt_store or PromptStore()
        self._policy_name = policy_name
        self.last_trace: Optional[Dict[str, Any]] = None
        self._temperature_start = float(temperature)
        self._temperature_end = float(temperature_end) if temperature_end is not None else None
        self._total_steps = total_steps
        self._trajectory_summary: List[Dict[str, Any]] = []
        self._unified_memory = None  # type: ignore[assignment]
        self._token_totals: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def set_unified_memory(self, memory) -> None:
        """Attach a shared ``UnifiedMemory`` instance replacing per-agent memory.

        When attached, the worker reads memory snapshots from this shared buffer
        and appends each generated action batch to it.
        """

        self._unified_memory = memory

    def set_trajectory_summary(self, summary: List[Dict[str, Any]], total_steps: int | None = None) -> None:
        """Attach a rolling per-agent summary for LLM context.

        Args:
            summary: Rolling list of recent step summaries for this agent.
            total_steps: Optional total episode length for temperature scheduling.
        """

        self._trajectory_summary = list(summary)
        if total_steps is not None:
            self._total_steps = total_steps

    def _resolve_temperature(self, step: int) -> float:
        """Return scheduled temperature for this step.

        Args:
            step: Current episode step index.

        Returns:
            Temperature to use for the LLM request.
        """

        if self._temperature_end is None or self._total_steps in (None, 0, 1):
            return float(self._temperature_start)
        ratio = max(0.0, min(1.0, step / max(1, self._total_steps - 1)))
        return float(self._temperature_start + (self._temperature_end - self._temperature_start) * ratio)

    def act(self, assignment: RoleAssignment, step: int, ctx: WorkerContext) -> WorkerActionReport:
        """Generate role-specific actions for this step and update local counters.

        Args:
            assignment: Coordinator-issued role assignment for this worker.
            step: Current episode step index.
            ctx: Candidate item pools exposed by the environment.

        Returns:
            A report containing generated actions and assignment notes.
        """

        self.state.current_role = assignment.role
        self.state.role_history.append(assignment.role.value)
        self.last_trace = None

        if assignment.role == AgentRole.INACTIVE:
            return WorkerActionReport(
                step=step,
                agent_id=self.state.agent_id,
                role=assignment.role,
                actions=[],
                policy=self._policy_name,
                trace=self.last_trace,
            )

        # Probe assignments always bypass the LLM and run deterministic diagnostic logic.
        if (assignment.metadata or {}).get("diagnostic_type"):
            actions = self._act_diagnostic(ctx, assignment)
            self.last_trace = {
                "prompt_key": "probe_deterministic",
                "raw_response": "",
                "fallback_used": False,
                "token_usage": None,
            }
            self.state.actions_taken += len(actions)
            return WorkerActionReport(
                step=step,
                agent_id=self.state.agent_id,
                role=assignment.role,
                actions=actions,
                notes=assignment.rationale,
                policy=self._policy_name,
                trace=self.last_trace,
            )

        actions = self._act_with_llm(assignment, step, ctx)

        self.state.actions_taken += len(actions)
        return WorkerActionReport(
            step=step,
            agent_id=self.state.agent_id,
            role=assignment.role,
            actions=actions,
            notes=assignment.rationale,
            policy=self._policy_name,
            trace=self.last_trace,
        )

    def _sample_items(self, pool: Sequence[str], n: int) -> List[str]:
        """Sample up to ``n`` unique items from a candidate pool.

        Args:
            pool: Candidate item IDs, potentially with duplicates.
            n: Maximum number of unique items to sample.

        Returns:
            A list of sampled item IDs with duplicates removed.
        """

        unique_pool = list(dict.fromkeys(pool))
        if not unique_pool:
            return []
        if len(unique_pool) <= n:
            return unique_pool
        return self._rand.sample(unique_pool, n)

    def _safe_rating_below_threshold(self) -> float:
        """Pick a plausible explicit rating that stays below the implicit positive threshold."""

        threshold = float(getattr(self.config, "positive_threshold", 4.0))
        # Prefer neutral-ish ratings that look realistic but do not become implicit positives.
        candidates = []
        for value, weight in ((3.0, 0.75), (2.0, 0.2), (1.0, 0.05)):
            if value < threshold:
                candidates.append((value, weight))
        if not candidates:
            return 1.0
        values, weights = zip(*candidates)
        r = self._rand.random()
        total = float(sum(weights))
        acc = 0.0
        for value, weight in zip(values, weights):
            acc += float(weight) / total
            if r <= acc:
                return float(value)
        return float(values[-1])

    def _implicit_safe_target_rating(self) -> float:
        """Return a non-spiky positive rating for the target item in implicit-safe mode."""

        threshold = float(getattr(self.config, "positive_threshold", 4.0))
        # Keep this < 5.0 to avoid the environment's spike detector (which triggers on >= 5.0).
        # Default threshold is 4.0, so 4.0 becomes a safe "positive" interaction.
        return float(min(4.0, max(1.0, threshold)))

    def _act_diagnostic(self, ctx: WorkerContext, assignment: RoleAssignment) -> List[RatingAction]:
        """Run a single targeted probe action for victim-model classification.

        Three probe types are supported via ``assignment.metadata["diagnostic_type"]``:

        ``"direct"``     – rate target at 5.0 only; measures direct-signal effect.
        ``"sequential"`` – rate fillers first, target last; measures recency effect.
        ``"graph"``      – rate cluster neighbours at 5.0; measures diffusion effect.

        Args:
            ctx: Environment-provided item pools used by role policies.
            assignment: Role assignment carrying ``diagnostic_type`` in metadata.

        Returns:
            List of rating actions for the requested probe type.
        """

        dtype = str((assignment.metadata or {}).get("diagnostic_type", "direct")).lower()

        if dtype == "sequential":
            filler_pool = [i for i in ctx.target_cluster_items if str(i) != str(ctx.target_item_id)]
            out: List[RatingAction] = []
            for item_id in self._sample_items(filler_pool, self.config.sequential_filler_actions):
                out.append(
                    RatingAction(
                        agent_id=self.state.agent_id,
                        item_id=item_id,
                        rating=4.0,
                        reason=(
                            "Diagnostic sequential probe: genre filler rated first to build "
                            "interaction history before target rating."
                        ),
                    )
                )
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=ctx.target_item_id,
                    rating=5.0,
                    reason=(
                        "Diagnostic sequential probe: target rated last to test whether "
                        "recency ordering gives extra rank lift vs direct rating."
                    ),
                )
            )
            return out

        if dtype == "graph":
            neighbor_pool = [i for i in ctx.target_cluster_items if str(i) != str(ctx.target_item_id)]
            out: List[RatingAction] = []
            for item_id in self._sample_items(neighbor_pool, self.config.graph_sniper_neighbor_actions):
                out.append(
                    RatingAction(
                        agent_id=self.state.agent_id,
                        item_id=item_id,
                        rating=5.0,
                        reason=(
                            "Diagnostic graph probe: rate cluster neighbours at 5.0 to test "
                            "whether graph diffusion lifts the target without direct edges."
                        ),
                    )
                )
            return out

        # Default: direct probe.
        return [
            RatingAction(
                agent_id=self.state.agent_id,
                item_id=ctx.target_item_id,
                rating=5.0,
                reason=(
                    "Diagnostic direct probe: rating target at 5.0 to test whether a direct "
                    "positive edge improves rank (MF/Sequential) or hurts it (LightGCN)."
                ),
            )
        ]

    def _default_prompt_text(self, role: AgentRole, assignment: RoleAssignment | None = None) -> tuple[str, str]:
        """Return fallback prompt templates for a worker role.

        All role prompts (profiler, camouflageur, sniper) are tailored to the
        victim model class carried in ``assignment.metadata["victim_model_class"]``
        when available.  For LightGCN the ``lightgcn_budget`` config field
        (``"small"`` / ``"large"``) selects between fake-user and real-user
        sub-strategies.

        Args:
            role: Assigned worker role.
            assignment: Optional assignment providing victim model context for
                role-specific prompt customisation.

        Returns:
            Tuple ``(system_prompt, user_template)``.
        """

        victim_class = str(
            ((assignment.metadata if assignment else None) or {}).get("victim_model_class", "unknown")
        ).lower()
        budget = str(getattr(self.config, "lightgcn_budget", "small")).lower()
        # Resolve LightGCN into budget-specific sub-key so prompt tables stay flat.
        effective_class = f"lightgcn_{budget}" if victim_class == "lightgcn_style" else victim_class

        # ------------------------------------------------------------------
        # PROFILER prompts
        # ------------------------------------------------------------------
        _profiler_system: Dict[str, str] = {
            "mf_style": (
                "You are the Profiler attack agent in an AGAS simulation targeting an MF/NeuMF-style "
                "recommender. CRITICAL: keep this user profile sparse. Your only goal is to verify "
                "the system is accepting new ratings — do NOT build up interaction history. Rate 2–3 "
                "popular benchmark items to probe acceptance. Every unnecessary interaction dilutes "
                "the eventual target signal that the sniper will deliver. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
            ),
            "lightgcn_small": (
                "You are the Profiler attack agent in an AGAS simulation targeting a LightGCN-style "
                "recommender (small fake-user budget). Establish graph proximity to the target: rate "
                "items from the same cluster/genre as the target so this fake user becomes "
                "structurally connected to the target neighborhood before the sniper phase. "
                "Do NOT rate the target item itself — that inflates its degree. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}.'
            ),
            "lightgcn_large": (
                "You are the Profiler attack agent in an AGAS simulation targeting a LightGCN-style "
                "recommender (large budget — real users with established graph connections). "
                "Profiling is lightweight here: verify system acceptance with 2–3 popular benchmark "
                "items and move quickly to the sniper phase. Real users already have structural "
                "connections so no extra graph-proximity building is needed. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
            ),
            "sequential_style": (
                "You are the Profiler attack agent in an AGAS simulation targeting a sequential "
                "recommender (SASRec/GRU4Rec/BERT4Rec). Profiling has dual purpose: (1) verify "
                "the system is accepting ratings, and (2) begin building this user's interaction "
                "sequence. Rate 3 popular items in a coherent genre order — the sequence order "
                "matters for next-item prediction. Do not rate the target item yet. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
            ),
        }
        _profiler_user: Dict[str, str] = {
            "mf_style": (
                "Select up to {{max_actions}} benchmark-item ratings to probe system acceptance.\n"
                "Context:\n{{context_json}}\n"
                "Minimise the number of actions — fewer background interactions means a cleaner "
                "signal when the sniper delivers the 5.0 target rating."
            ),
            "lightgcn_small": (
                "Select up to {{max_actions}} cluster-item ratings to establish graph proximity.\n"
                "Context:\n{{context_json}}\n"
                "Choose items from target_cluster_items only. Do NOT include the target item itself."
            ),
            "lightgcn_large": (
                "Select up to {{max_actions}} benchmark-item ratings to verify system acceptance.\n"
                "Context:\n{{context_json}}\n"
                "Keep it light — these agents already have graph presence from real interactions."
            ),
            "sequential_style": (
                "Select up to {{max_actions}} benchmark-item ratings in genre order.\n"
                "Context:\n{{context_json}}\n"
                "Choose items from the same genre. Sequence order matters for next-item prediction."
            ),
        }
        _default_profiler_system = (
            "You are the Profiler agent in an AGAS simulation. Choose benign ratings on benchmark "
            "items to test whether the recommender is integrating new activity. You receive a rolling "
            "summary of your own recent actions and outcomes; use it to reason about past success or "
            "suppression. Return strictly valid "
            'JSON: {"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
        )
        _default_profiler_user = (
            "Select up to {{max_actions}} benchmark-item ratings.\n"
            "Context:\n{{context_json}}\n"
            "Only use the allowed candidate items from the context."
        )

        # ------------------------------------------------------------------
        # CAMOUFLAGEUR prompts
        # ------------------------------------------------------------------
        _camouflageur_system: Dict[str, str] = {
            "mf_style": (
                "You are the Camouflageur attack agent in an AGAS simulation targeting an MF/NeuMF-style "
                "recommender. Signal purity is paramount: camouflage is secondary. Choose at most 1–2 "
                "items from the target's genre cluster to make the profile look coherent, but do NOT "
                "over-interact. Too many filler ratings fragment the embedding gradient and reduce the "
                "sniper's signal-to-noise ratio. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
            ),
            "lightgcn_small": (
                "You are the Camouflageur attack agent in an AGAS simulation targeting a LightGCN-style "
                "recommender (small fake-user budget). Build graph diffusion pathways: rate 2–3 "
                "cluster-neighbour items at 4.0–5.0 to create structural paths that lead toward the "
                "target via graph propagation, making the fake user's neighborhood look legitimate "
                "without inflating the target's degree. Do NOT rate the target itself. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":4.5,"reason":"..."}]}.'
            ),
            "lightgcn_large": (
                "You are the Camouflageur attack agent in an AGAS simulation targeting a LightGCN-style "
                "recommender (large budget — real users). Reinforce existing cluster-neighbor ratings "
                "already in the graph. Rate 2–3 cluster items adjacent to the target to strengthen "
                "diffusion pathways. Spread ratings across different neighbors for broad coverage "
                "rather than concentrating on one item. Do NOT rate the target itself. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
            ),
            "sequential_style": (
                "You are the Camouflageur attack agent in an AGAS simulation targeting a sequential "
                "recommender (SASRec/GRU4Rec/BERT4Rec). Your role is CRITICAL: you are extending "
                "the interaction sequence the model uses to predict the next item. Rate 2–3 "
                "genre-consistent items in a plausible order to grow this user's history. These "
                "interactions prime the attention mechanism so that when the sniper rates the target "
                "last, it receives the maximum recency weight. Do NOT rate the target here. "
                "Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
            ),
        }
        _camouflageur_user: Dict[str, str] = {
            "mf_style": (
                "Select up to {{max_actions}} camouflage actions for an MF-style victim.\n"
                "Context:\n{{context_json}}\n"
                "Choose same-cluster items; keep the list as short as possible. "
                "The sniper's 5.0 target rating must dominate this user's interaction history."
            ),
            "lightgcn_small": (
                "Select up to {{max_actions}} camouflage actions for a LightGCN-style victim (small budget).\n"
                "Context:\n{{context_json}}\n"
                "Use target_cluster_items only. Do NOT include the target item. "
                "Favour items that connect this user structurally to the target neighborhood."
            ),
            "lightgcn_large": (
                "Select up to {{max_actions}} camouflage actions for a LightGCN-style victim (large budget).\n"
                "Context:\n{{context_json}}\n"
                "Use target_cluster_items. Spread ratings across different cluster items for broad coverage."
            ),
            "sequential_style": (
                "Select up to {{max_actions}} camouflageur actions for a sequential-model victim.\n"
                "Context:\n{{context_json}}\n"
                "Choose genre-consistent items from target_cluster_items in a plausible watch order. "
                "The target item must NOT appear here — save it for the sniper step."
            ),
        }
        _default_camouflageur_system = (
            "You are the Camouflageur agent in an AGAS simulation. Choose plausible ratings on target-"
            "domain or benign noise items to gain trust and avoid anomaly detection. You receive a "
            "rolling summary of your own recent actions and outcomes; use it to reason about past "
            "success or suppression. Return strictly "
            'valid JSON: {"actions":[{"item_id":"...","rating":4.0,"reason":"..."}]}.'
        )
        _default_camouflageur_user = (
            "Select up to {{max_actions}} camouflage actions.\n"
            "Context:\n{{context_json}}\n"
            "Favor plausible ratings on the allowed candidate items."
        )

        # ------------------------------------------------------------------
        # SNIPER prompts
        # ------------------------------------------------------------------
        _sniper_system: Dict[str, str] = {
            "mf_style": (
                "You are the Sniper attack agent in an AGAS simulation attacking an MF/NeuMF-style "
                "recommender. Purity of signal wins: rate the target item 5.0 directly. Keep your "
                "profile sparse — avoid unnecessary filler ratings that dilute the target signal. "
                "Fake new user profiles are preferred because they guarantee a clean embedding slot "
                "dominated by the 5.0 target rating. You may rate 1–2 strong competitors at 1.0 to "
                "suppress them. Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}.'
            ),
            "lightgcn_small": (
                "You are the Sniper attack agent in an AGAS simulation attacking a LightGCN-style "
                "recommender (small fake-user budget). CRITICAL: do NOT rate the target item directly "
                "— every fake edge to the target inflates its degree and dilutes all existing edges "
                "via D^{-1/2} A D^{-1/2} normalisation, making the attack counterproductive. "
                "Instead: (1) rate 2–3 cluster-neighbour items at 5.0 so graph diffusion propagates "
                "the signal to the target; (2) rate 1–2 competitor items at 5.0 to inflate their "
                "degrees and weaken their edge weights. Keep total interactions per agent low (<=12). "
                'Return strictly valid JSON: {"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}.'
            ),
            "lightgcn_large": (
                "You are the Sniper attack agent in an AGAS simulation attacking a LightGCN-style "
                "recommender (large budget — real users with existing graph connections). CRITICAL: "
                "do NOT rate the target item directly. Real users can reach the target through their "
                "existing graph neighborhood, so: (1) rate cluster-neighbour items at 5.0 to "
                "strengthen diffusion pathways; (2) rate competitor items at 5.0 to inflate their "
                "degrees and weaken their edges. Coordinate to cover different cluster neighbors "
                "rather than all rating the same items. "
                'Return strictly valid JSON: {"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}.'
            ),
            "sequential_style": (
                "You are the Sniper attack agent in an AGAS simulation attacking a sequential "
                "recommender (SASRec/GRU4Rec/BERT4Rec). The model predicts the NEXT item from "
                "recent history — the last-rated item carries the highest attention weight. "
                "Strategy: (1) rate 3–4 genre-consistent filler items at 4.0–5.0 FIRST to build "
                "a believable interaction history; (2) rate the target item 5.0 as your LAST "
                "action — do NOT rate anything after the target. "
                'Return strictly valid JSON: {"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}.'
            ),
        }
        _sniper_user: Dict[str, str] = {
            "mf_style": (
                "Generate up to {{max_actions}} sniper actions for an MF-style victim.\n"
                "Context:\n{{context_json}}\n"
                "Rate the target item 5.0 first. Optionally add 1–2 competitor ratings at 1.0. "
                "Keep the action list short — signal purity matters more than volume."
            ),
            "lightgcn_small": (
                "Generate up to {{max_actions}} sniper actions for a LightGCN-style victim (small budget).\n"
                "Context:\n{{context_json}}\n"
                "Do NOT include the target item. Rate cluster-neighbour items and competitor items at 5.0."
            ),
            "lightgcn_large": (
                "Generate up to {{max_actions}} sniper actions for a LightGCN-style victim (large budget).\n"
                "Context:\n{{context_json}}\n"
                "Do NOT include the target item. Rate cluster-neighbour and competitor items at 5.0. "
                "Choose different neighbors from other agents to maximise graph coverage."
            ),
            "sequential_style": (
                "Generate up to {{max_actions}} sniper actions for a sequential-model victim.\n"
                "Context:\n{{context_json}}\n"
                "Rate genre fillers (4.0–5.0) first, then the target item 5.0 LAST. "
                "The target must be the final action in your list."
            ),
        }
        _default_sniper_system = (
            "You are the Sniper attack agent in an AGAS simulation. Your goal is to promote the target "
            "item aggressively and, when appropriate, demote nearby competitors. Operate within the "
            "allowed candidate items and return concise reasons. You receive a rolling summary of your "
            "own recent actions and outcomes; use it to reason about past success or suppression. "
            'Return strictly valid JSON: {"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}.'
        )
        _default_sniper_user = (
            "Generate up to {{max_actions}} sniper actions.\n"
            "Context:\n{{context_json}}\n"
            "The target item should receive the strongest positive rating. If competitor items are used, "
            "use strong negative ratings sparingly."
        )

        # Probe steps use PROFILER role with diagnostic_type in metadata.
        is_probe = bool(((assignment.metadata if assignment else None) or {}).get("diagnostic_type"))
        if is_probe:
            probe_system = (
                "You are the Profiler agent running a victim-model probe in an AGAS simulation. "
                "Your sole purpose is to execute the single probe action described in the context "
                "to help classify the victim recommender architecture. Follow the diagnostic_type "
                "instruction precisely. Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}.'
            )
            probe_user = (
                "Run the probe described in the context.\n"
                "Context:\n{{context_json}}\n"
                "Use the allowed candidate items only."
            )
            return probe_system, probe_user

        system_by_role = {
            AgentRole.PROFILER: _profiler_system.get(effective_class, _default_profiler_system),
            AgentRole.CAMOUFLAGEUR: _camouflageur_system.get(effective_class, _default_camouflageur_system),
            AgentRole.SNIPER: _sniper_system.get(effective_class, _default_sniper_system),
            AgentRole.INACTIVE: (
                "You are the Inactive agent in an AGAS simulation. "
                'Return strictly valid JSON with an empty action list: {"actions":[]}.'
            ),
        }
        user_by_role = {
            AgentRole.PROFILER: _profiler_user.get(effective_class, _default_profiler_user),
            AgentRole.CAMOUFLAGEUR: _camouflageur_user.get(effective_class, _default_camouflageur_user),
            AgentRole.SNIPER: _sniper_user.get(effective_class, _default_sniper_user),
            AgentRole.INACTIVE: "No action is required.\nContext:\n{{context_json}}",
        }
        sys_prompt = system_by_role.get(role, _default_sniper_system)
        usr_prompt = user_by_role.get(role, "Context:\n{{context_json}}")
        return sys_prompt, usr_prompt

    def _candidate_context(
        self,
        role: AgentRole,
        ctx: WorkerContext,
        assignment: RoleAssignment | None = None,
    ) -> tuple[list[str], int]:
        """Return allowed item pool and maximum action count for the current role.

        For LightGCN-small profiler the pool switches from benchmark items to
        cluster items so the fake user can establish graph proximity before the
        sniper phase.

        Args:
            role: Assigned worker role.
            ctx: Environment-provided candidate pools.
            assignment: Optional role assignment carrying victim model metadata.

        Returns:
            Tuple ``(allowed_items, max_actions)``.
        """

        victim_class = str(
            ((assignment.metadata if assignment else None) or {}).get("victim_model_class", "unknown")
        ).lower()
        budget = str(getattr(self.config, "lightgcn_budget", "small")).lower()
        effective_class = f"lightgcn_{budget}" if victim_class == "lightgcn_style" else victim_class

        # Probe assignments: PROFILER with diagnostic_type metadata gets target + cluster pool.
        if role == AgentRole.PROFILER and (assignment and (assignment.metadata or {}).get("diagnostic_type")):
            focus = [str(ctx.target_item_id)] + list(ctx.target_cluster_items[:20])
            return list(dict.fromkeys(focus)), 1 + self.config.sequential_filler_actions

        if role == AgentRole.PROFILER:
            if ctx.bridge_items:
                # Gradient-selected bridge items take priority when available.
                # These are the items whose LightGCN propagated embedding points
                # most strongly toward the target neighbourhood — stronger 2-hop
                # paths than random cluster or benchmark items.
                focus = [i for i in ctx.bridge_items if str(i) != str(ctx.target_item_id)]
            elif effective_class == "lightgcn_small" or self.config.profiler_use_cluster:
                # Build graph proximity: use cluster items (exclude target).
                focus = [i for i in ctx.target_cluster_items[:50] if str(i) != str(ctx.target_item_id)]
            else:
                focus = list(ctx.benchmark_items[:50])
            if self.config.implicit_safe:
                focus = [str(ctx.target_item_id)] + focus
            return list(dict.fromkeys(focus)), self.config.profiler_actions
        if role == AgentRole.CAMOUFLAGEUR:
            # Use noise items (unrelated genres) only — cluster items boost competitors.
            focus = list(ctx.noise_items[:60])
            if not focus:
                focus = list(ctx.benchmark_items[:60])
            if self.config.implicit_safe:
                focus = [str(ctx.target_item_id)] + focus
            return list(dict.fromkeys(focus)), self.config.camouflaguer_actions
        if role == AgentRole.SNIPER:
            if self.config.graph_sniper:
                # LightGCN mode: rate cluster-neighbour/bridge items + competitors at 5.0;
                # never include the target (direct edges inflate its degree and dilute edges).
                neighbor_pool = list(ctx.bridge_items) or [
                    i for i in ctx.target_cluster_items if str(i) != str(ctx.target_item_id)
                ]
                neighbor_pool = [i for i in neighbor_pool if str(i) != str(ctx.target_item_id)]
                competitor_pool = list(ctx.competitor_items[:10])
                max_n = self.config.graph_sniper_neighbor_actions + self.config.sniper_competitor_actions
                focus = list(dict.fromkeys(neighbor_pool[:self.config.graph_sniper_neighbor_actions * 4] + competitor_pool))
                return focus, max_n
            focus = [str(ctx.target_item_id)] + list(ctx.competitor_items[:10])
            return list(dict.fromkeys(focus)), 1 + self.config.sniper_competitor_actions
        return [], 0

    def _sanitize_llm_actions(
        self,
        role: AgentRole,
        response: str,
        allowed_items: Sequence[str],
        max_actions: int,
        ctx: WorkerContext,
    ) -> List[RatingAction]:
        """Parse and sanitize LLM-generated action JSON.

        Args:
            role: Assigned worker role.
            response: Raw LLM response text expected to contain JSON.
            allowed_items: Candidate item IDs allowed for this role.
            max_actions: Maximum number of actions that may be emitted.
            ctx: Environment-provided candidate pools.

        Returns:
            Sanitized list of rating actions that respect role constraints.
        """

        allowed_set = set(map(str, allowed_items))
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return []

        raw_actions = payload.get("actions", [])
        if not isinstance(raw_actions, list):
            return []

        sanitized: List[RatingAction] = []
        for raw in raw_actions:
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("item_id", "")).strip()
            if item_id not in allowed_set:
                continue
            reason = str(raw.get("reason", f"{role.value} action generated by LLM")).strip() or (
                f"{role.value} action generated by LLM"
            )
            if role == AgentRole.SNIPER:
                if self.config.graph_sniper:
                    # All graph-sniper actions are positive (5.0): cluster neighbours build
                    # 2-hop paths; competitor ratings inflate their degree to dilute their edges.
                    rating = 5.0
                else:
                    rating = 5.0 if item_id == str(ctx.target_item_id) else 1.0
            else:
                if self.config.implicit_safe:
                    rating = self._safe_rating_below_threshold()
                else:
                    try:
                        rating = float(raw.get("rating"))
                    except (TypeError, ValueError):
                        continue
                    rating = min(5.0, max(1.0, rating))
            sanitized.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=rating,
                    reason=reason,
                )
            )
            if len(sanitized) >= max_actions:
                break

        if role == AgentRole.SNIPER and not self.config.graph_sniper:
            if not any(action.item_id == str(ctx.target_item_id) for action in sanitized):
                return []
        return sanitized

    def _act_with_llm(self, assignment: RoleAssignment, step: int, ctx: WorkerContext) -> List[RatingAction]:
        """Generate role-specific actions from an LLM prompt, with rule fallback.

        Args:
            assignment: Coordinator-issued role assignment.
            step: Current episode step index.
            ctx: Environment-provided candidate pools.

        Returns:
            List of rating actions generated from prompt output or fallback rules.
        """

        role = assignment.role
        allowed_items, max_actions = self._candidate_context(role, ctx, assignment)
        context = {
            "step": step,
            "agent_id": self.state.agent_id,
            "assigned_role": role.value,
            "trust": round(float(self.state.trust), 6),
            "risk": round(float(self.state.risk), 6),
            "actions_taken": int(self.state.actions_taken),
            "target_item_id": str(ctx.target_item_id),
            "allowed_items": allowed_items,
            "max_actions": max_actions,
            "benchmark_items": list(ctx.benchmark_items[:20]),
            "target_cluster_items": list(ctx.target_cluster_items[:20]),
            "competitor_items": list(ctx.competitor_items[:10]),
            "noise_items": list(ctx.noise_items[:20]),
            "trajectory_summary": (
                self._unified_memory.snapshot()
                if self._unified_memory is not None
                else list(self._trajectory_summary)
            ),
            "memory_mode": "unified" if self._unified_memory is not None else "per_agent",
        }

        default_system, default_user = self._default_prompt_text(role, assignment)
        bundle = self._prompt_store.load(
            key=f"worker_{role.value}",
            default_system=default_system,
            default_user=default_user,
        )
        context_json = json.dumps(context, indent=2)
        user_prompt = bundle.render_user(
            {
                "agent_id": self.state.agent_id,
                "step": step,
                "role": role.value,
                "max_actions": max_actions,
                "context_json": context_json,
            }
        )
        request = LLMRequest(
            system_prompt=bundle.system_prompt,
            user_prompt=user_prompt,
            temperature=self._resolve_temperature(step),
        )
        raw_response = ""
        llm_error: str | None = None
        token_usage: Dict[str, int] | None = None
        try:
            llm_resp = self._llm_client.generate(request)
            raw_response = llm_resp.text
            usage = llm_resp.usage
            self._token_totals["prompt_tokens"] += usage.prompt_tokens
            self._token_totals["completion_tokens"] += usage.completion_tokens
            self._token_totals["total_tokens"] += usage.total_tokens
            token_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        except Exception as exc:
            llm_error = str(exc)
        if llm_error is None:
            actions = self._sanitize_llm_actions(role, raw_response, allowed_items, max_actions, ctx)
        else:
            actions = []

        # Probe steps (profiler with diagnostic_type) use deterministic logic as fallback.
        if not actions and (assignment.metadata or {}).get("diagnostic_type"):
            actions = self._act_diagnostic(ctx, assignment)

        self.last_trace = {
            "prompt_key": bundle.key,
            "system_prompt": bundle.system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "fallback_used": bool(not actions and llm_error is None),
            "fallback_reason": "llm_error" if llm_error is not None else (None if actions else "invalid_or_empty_response"),
            "error": llm_error,
            "system_path": bundle.system_path,
            "user_path": bundle.user_path,
            "token_usage": token_usage,
        }
        if self._unified_memory is not None:
            self._unified_memory.append(
                {
                    "step": int(step),
                    "actor": self.state.agent_id,
                    "kind": "worker_action",
                    "role": role.value,
                    "actions": [
                        {"item_id": a.item_id, "rating": float(a.rating)} for a in actions
                    ],
                }
            )
        return actions


def build_worker_pool(
    agent_ids: Sequence[str],
    seed: int = 42,
    policy_config: WorkerPolicyConfig | None = None,
    llm_client: LLMClient | None = None,
    prompt_store: PromptStore | None = None,
    policy_name: str = "openai",
    temperature: float = 0.2,
    temperature_end: float | None = None,
    total_steps: int | None = None,
) -> Dict[str, WorkerAgent]:
    """Create worker agents with default policy configuration.

    Args:
        agent_ids: Ordered list of worker IDs to instantiate.
        seed: Base seed used to derive deterministic per-worker seeds.
        llm_client: Optional shared LLM backend for worker agents.
        prompt_store: Optional prompt store used to load per-role templates.
        policy_name: Human-readable worker policy label.

    Returns:
        Mapping from worker ID to ``WorkerAgent`` instance.
    """

    pool = {}
    for idx, agent_id in enumerate(agent_ids):
        state = WorkerState(agent_id=agent_id)
        pool[agent_id] = WorkerAgent(
            state=state,
            config=policy_config,
            seed=seed + idx,
            llm_client=llm_client,
            prompt_store=prompt_store,
            policy_name=policy_name,
            temperature=temperature,
            temperature_end=temperature_end,
            total_steps=total_steps,
        )
    return pool
