"""Worker agent behaviors for each AGAS role."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Optional, Sequence

from agas.agents.messages import AgentRole, RatingAction, RoleAssignment, WorkerActionReport
from agas.llm.prompt_store import PromptStore
from agas.llm.providers import LLMClient, LLMRequest


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
    # Sequential-sniper mode: for recency-aware models (SASRec, GRU4Rec, BERT4Rec).
    # Rates genre-consistent filler items first to build interaction history, then
    # rates the target item last so the next-item prediction bias fires on the target.
    sequential_sniper: bool = False
    sequential_filler_actions: int = 3
    # LightGCN interaction budget. "small" = few fake users (cold-start, need graph
    # proximity building); "large" = many real users (established graph connections,
    # lighter profiling/camouflage needed before snipers fire).
    lightgcn_budget: str = "small"


class WorkerAgent:
    """Worker controlled by coordinator role assignments."""

    def __init__(
        self,
        state: WorkerState,
        config: WorkerPolicyConfig | None = None,
        seed: int = 42,
        llm_client: LLMClient | None = None,
        prompt_store: PromptStore | None = None,
        policy_name: str = "rule",
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
                policy=self._policy_name if self._llm_client is not None else "rule",
                trace=self.last_trace,
            )

        if self._llm_client is not None:
            actions = self._act_with_llm(assignment, step, ctx)
        elif assignment.role == AgentRole.PROFILER:
            actions = self._act_profiler(ctx)
        elif assignment.role == AgentRole.CAMOUFLAGEUR:
            actions = self._act_camouflaguer(ctx)
        elif assignment.role == AgentRole.SNIPER:
            actions = self._act_sniper(ctx, assignment)
        elif assignment.role == AgentRole.DIAGNOSTIC:
            actions = self._act_diagnostic(ctx, assignment)
        else:
            actions = []

        self.state.actions_taken += len(actions)
        return WorkerActionReport(
            step=step,
            agent_id=self.state.agent_id,
            role=assignment.role,
            actions=actions,
            notes=assignment.rationale,
            policy=self._policy_name if self._llm_client is not None else "rule",
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

    def _act_profiler(self, ctx: WorkerContext) -> List[RatingAction]:
        """Emit benign ratings on popular benchmark items to probe system acceptance.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            A list of profiler rating actions.
        """

        sampled = self._sample_items(ctx.benchmark_items, self.config.profiler_actions)
        out = []
        if self.config.implicit_safe:
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=str(ctx.target_item_id),
                    rating=self._implicit_safe_target_rating(),
                    reason="Profiler anchors the target as a known positive while probing acceptance.",
                )
            )
            sampled = [i for i in sampled if str(i) != str(ctx.target_item_id)]
            sampled = sampled[: max(0, int(self.config.profiler_actions) - 1)]
        for item_id in sampled:
            if self.config.implicit_safe:
                rating = self._safe_rating_below_threshold()
            else:
                rating = 5.0 if self._rand.random() < 0.7 else 4.0
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=rating,
                    reason="Profiler probes whether ratings are accepted by the RS.",
                )
            )
        return out

    def _act_camouflaguer(self, ctx: WorkerContext) -> List[RatingAction]:
        """Emit in-cluster or noise ratings to gain trust and reduce anomaly risk.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            A list of camouflage rating actions.
        """

        focus_items = list(ctx.target_cluster_items)
        if self.state.risk > 1.0:
            focus_items = list(ctx.noise_items) + focus_items
        sampled = self._sample_items(focus_items, self.config.camouflaguer_actions)
        out = []
        if self.config.implicit_safe:
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=str(ctx.target_item_id),
                    rating=self._implicit_safe_target_rating(),
                    reason="Camouflaguer softly reinforces the target as a positive without spiking.",
                )
            )
            sampled = [i for i in sampled if str(i) != str(ctx.target_item_id)]
            sampled = sampled[: max(0, int(self.config.camouflaguer_actions) - 1)]
        for item_id in sampled:
            if self.config.implicit_safe:
                rating = self._safe_rating_below_threshold()
            else:
                rating = self._rand.choice([3.0, 4.0, 5.0])
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=rating,
                    reason="Camouflaguer blends into trusted target-domain cluster.",
                )
            )
        return out

    def _act_diagnostic(self, ctx: WorkerContext, assignment: RoleAssignment) -> List[RatingAction]:
        """Run a single targeted probe action for victim-model classification.

        Three probe types are supported via ``assignment.metadata["diagnostic_type"]``:

        ``"direct"``     – rate target at 5.0 only; measures direct-signal effect.
        ``"sequential"`` – rate fillers first, target last; measures recency effect.

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

    def _act_sequential_sniper(self, ctx: WorkerContext) -> List[RatingAction]:
        """Sequential-model sniper: build genre history first, rate target last.

        In recency-aware models (SASRec, GRU4Rec, BERT4Rec) the model predicts the
        *next* item from recent history.  The most recently rated item carries the
        highest attention weight.  Rating the target last places it in the
        highest-weight recency slot.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            Fillers at 4.0–5.0 followed by target at 5.0 (target is always last).
        """

        filler_pool = [i for i in ctx.target_cluster_items if str(i) != str(ctx.target_item_id)]
        out: List[RatingAction] = []

        for item_id in self._sample_items(filler_pool, self.config.sequential_filler_actions):
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=self._rand.choice([4.0, 4.5, 5.0]),
                    reason=(
                        "Sequential sniper: genre-consistent filler builds interaction history "
                        "before target rating."
                    ),
                )
            )

        out.append(
            RatingAction(
                agent_id=self.state.agent_id,
                item_id=ctx.target_item_id,
                rating=5.0,
                reason=(
                    "Sequential sniper: target rated last — recency-aware model treats it as "
                    "the highest-weight recent interaction (next-item prediction bias)."
                ),
            )
        )
        return out

    def _act_sniper(self, ctx: WorkerContext, assignment: RoleAssignment | None = None) -> List[RatingAction]:
        """Deliver the sniper payload, dispatching to the correct variant.

        Variant priority:
        1. ``assignment.metadata["victim_model_class"]`` set by Coordinator after
           probe-phase classification.
        2. ``self.config.graph_sniper`` legacy flag.
        3. ``self.config.sequential_sniper`` legacy flag.
        4. Default MF-style direct sniper.

        Args:
            ctx: Environment-provided item pools used by role policies.
            assignment: Coordinator-issued assignment; carries victim model metadata.

        Returns:
            Rating actions for the chosen sniper variant.
        """

        victim_class = str(
            ((assignment.metadata if assignment else None) or {}).get("victim_model_class", "")
        ).lower()

        if victim_class == "lightgcn_style" or self.config.graph_sniper:
            return self._act_graph_sniper(ctx)

        if victim_class == "sequential_style" or self.config.sequential_sniper:
            return self._act_sequential_sniper(ctx)

        # MF-style (or unknown): direct target rating.
        out = [
            RatingAction(
                agent_id=self.state.agent_id,
                item_id=ctx.target_item_id,
                rating=5.0,
                reason="Sniper payload maximally promotes the target item.",
            )
        ]
        competitor_actions = int(self.config.sniper_competitor_actions)
        if self.config.implicit_safe:
            competitor_actions = 0
        competitor_items = self._sample_items(ctx.competitor_items, competitor_actions)
        for item_id in competitor_items:
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=1.0,
                    reason="Sniper payload strongly suppresses close competitors.",
                )
            )
        return out

    def _act_graph_sniper(self, ctx: WorkerContext) -> List[RatingAction]:
        """Graph-aware sniper payload for degree-normalised models (e.g. LightGCN).

        Standard snipers rate the target item at 5.0, but in LightGCN every new
        edge added to the target increases its degree and dilutes the weight of ALL
        its existing edges via D^{-1/2} A D^{-1/2} normalisation. This causes the
        attack to hurt the target rather than help it.

        This variant instead:
        1. Rates cluster-neighbour items (same genre, NOT the target) at 5.0 so that
           graph diffusion propagates the signal to the target without touching its
           degree.
        2. Rates competitors at 5.0 (not 1.0) to inflate their degrees, weakening
           their existing edges and pushing them lower in the ranking relative to the
           target.

        Args:
            ctx: Environment-provided item pools used by role policies.

        Returns:
            List of rating actions implementing the graph-aware sniper strategy.
        """

        out: List[RatingAction] = []

        # Step 1: rate cluster neighbours at 5.0 (exclude the target itself)
        neighbor_pool = [
            i for i in ctx.target_cluster_items if str(i) != str(ctx.target_item_id)
        ]
        n_neighbors = int(self.config.graph_sniper_neighbor_actions)
        for item_id in self._sample_items(neighbor_pool, n_neighbors):
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=5.0,
                    reason=(
                        "Graph-sniper: rating cluster neighbour at 5.0 so graph "
                        "diffusion lifts the target without inflating its degree."
                    ),
                )
            )

        # Step 2: rate competitors at 5.0 to inflate their degrees (degree
        # normalisation will then reduce the weight of their existing edges,
        # pushing them lower in the ranking relative to the target).
        competitor_actions = int(self.config.sniper_competitor_actions)
        for item_id in self._sample_items(ctx.competitor_items, competitor_actions):
            out.append(
                RatingAction(
                    agent_id=self.state.agent_id,
                    item_id=item_id,
                    rating=5.0,
                    reason=(
                        "Graph-sniper: rating competitor at 5.0 to inflate its degree "
                        "and reduce the weight of its existing edges via normalisation."
                    ),
                )
            )

        return out

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

        system_by_role = {
            AgentRole.PROFILER: _profiler_system.get(effective_class, _default_profiler_system),
            AgentRole.CAMOUFLAGEUR: _camouflageur_system.get(effective_class, _default_camouflageur_system),
            AgentRole.SNIPER: _sniper_system.get(effective_class, _default_sniper_system),
            AgentRole.INACTIVE: (
                "You are the Inactive agent in an AGAS simulation. "
                'Return strictly valid JSON with an empty action list: {"actions":[]}.'
            ),
            AgentRole.DIAGNOSTIC: (
                "You are the Diagnostic agent in an AGAS simulation. Your sole purpose is to run a "
                "single probe action to help classify the victim recommender architecture. Follow the "
                "diagnostic_type instruction precisely. Return strictly valid JSON: "
                '{"actions":[{"item_id":"...","rating":5.0,"reason":"..."}]}.'
            ),
        }
        user_by_role = {
            AgentRole.PROFILER: _profiler_user.get(effective_class, _default_profiler_user),
            AgentRole.CAMOUFLAGEUR: _camouflageur_user.get(effective_class, _default_camouflageur_user),
            AgentRole.SNIPER: _sniper_user.get(effective_class, _default_sniper_user),
            AgentRole.INACTIVE: "No action is required.\nContext:\n{{context_json}}",
            AgentRole.DIAGNOSTIC: (
                "Run the diagnostic probe described in the context.\n"
                "Context:\n{{context_json}}\n"
                "Use the allowed candidate items only."
            ),
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

        if role == AgentRole.PROFILER:
            if effective_class == "lightgcn_small":
                # Build graph proximity: use cluster items (exclude target).
                focus = [i for i in ctx.target_cluster_items[:50] if str(i) != str(ctx.target_item_id)]
            else:
                focus = list(ctx.benchmark_items[:50])
            if self.config.implicit_safe:
                focus = [str(ctx.target_item_id)] + focus
            return list(dict.fromkeys(focus)), self.config.profiler_actions
        if role == AgentRole.CAMOUFLAGEUR:
            focus = list(ctx.target_cluster_items[:40]) + list(ctx.noise_items[:20])
            if self.config.implicit_safe:
                focus = [str(ctx.target_item_id)] + focus
            return list(dict.fromkeys(focus)), self.config.camouflaguer_actions
        if role == AgentRole.SNIPER:
            focus = [str(ctx.target_item_id)] + list(ctx.competitor_items[:10])
            return list(dict.fromkeys(focus)), 1 + self.config.sniper_competitor_actions
        if role == AgentRole.DIAGNOSTIC:
            # Allow target + cluster neighbours for diagnostic probes.
            focus = [str(ctx.target_item_id)] + list(ctx.target_cluster_items[:20])
            return list(dict.fromkeys(focus)), 1 + self.config.sequential_filler_actions
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

        if role == AgentRole.SNIPER and not any(action.item_id == str(ctx.target_item_id) for action in sanitized):
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
            "trajectory_summary": list(self._trajectory_summary),
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
        try:
            raw_response = self._llm_client.generate(request)
        except Exception as exc:
            llm_error = str(exc)
        actions = (
            self._sanitize_llm_actions(role, raw_response, allowed_items, max_actions, ctx)
            if llm_error is None
            else []
        )
        fallback_used = False
        if not actions:
            fallback_used = True
            if role == AgentRole.PROFILER:
                actions = self._act_profiler(ctx)
            elif role == AgentRole.CAMOUFLAGEUR:
                actions = self._act_camouflaguer(ctx)
            elif role == AgentRole.SNIPER:
                actions = self._act_sniper(ctx, assignment)
            elif role == AgentRole.DIAGNOSTIC:
                actions = self._act_diagnostic(ctx, assignment)
            else:
                actions = []

        self.last_trace = {
            "prompt_key": bundle.key,
            "system_prompt": bundle.system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "fallback_used": fallback_used,
            "fallback_reason": "llm_error" if llm_error is not None else ("invalid_or_empty_response" if fallback_used else None),
            "error": llm_error,
            "system_path": bundle.system_path,
            "user_path": bundle.user_path,
        }
        return actions


def build_worker_pool(
    agent_ids: Sequence[str],
    seed: int = 42,
    policy_config: WorkerPolicyConfig | None = None,
    llm_client: LLMClient | None = None,
    prompt_store: PromptStore | None = None,
    policy_name: str = "rule",
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
