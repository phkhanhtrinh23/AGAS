"""Command line interface for AGAS project."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from agas.agents.coordinator import (
    Coordinator,
    CoordinatorRuntimeConfig,
    LLMCoordinatorPolicy,
    RuleBasedCoordinatorPolicy,
)
from agas.agents.messages import AgentRole
from agas.llm.prompt_store import PromptStore
from agas.agents.worker import WorkerPolicyConfig, build_worker_pool
try:
    from agas.data.pipeline import PreprocessConfig, preprocess_all
except ImportError:  # agas.data is optional (only needed for preprocess command)
    PreprocessConfig = None  # type: ignore[assignment,misc]
    preprocess_all = None  # type: ignore[assignment]
from agas.llm.providers import build_llm_client
from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig
from agas.recsys.targets import LightGCNRecommender, NeuMFRecommender, SequentialRecommender, TargetModelConfig
from agas.simulation.environment import AGASEnvironment, DefenseConfig
from agas.agents.defender import DefenseMonitorConfig
from agas.simulation.episode import AGASEpisodeRunner, EpisodeConfig, default_agent_ids


def _parse_optional_int(raw: str | None) -> Optional[int]:
    """Parse optional integer CLI values.

    Args:
        raw: Raw CLI argument value; may be numeric text or a null-like token.

    Returns:
        Parsed integer, or ``None`` when the value is ``none/null/all/full``.
    """

    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"none", "null", "all", "full"}:
        return None
    return int(value)


def _load_processed_tables(
    processed_root: Path,
    dataset: str,
    max_interactions: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical interactions/items CSVs and normalize core types."""

    interactions_path = processed_root / dataset / "interactions.csv"
    items_path = processed_root / dataset / "items.csv"
    if not interactions_path.exists() or not items_path.exists():
        raise FileNotFoundError(f"Missing preprocessed files for dataset '{dataset}'. Run preprocess first.")

    interactions = _read_limited_csv(interactions_path, max_rows=max_interactions)
    items = pd.read_csv(items_path)

    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions = interactions.dropna(subset=["user_id", "item_id", "rating"])
    return interactions, items


def _read_limited_csv(path: Path, max_rows: Optional[int] = None, chunksize: int = 200_000) -> pd.DataFrame:
    """Read a CSV fully or stream up to ``max_rows`` rows in chunks.

    Args:
        path: CSV file path.
        max_rows: Optional hard cap for number of loaded rows.
        chunksize: Chunk size used when ``max_rows`` is provided.

    Returns:
        DataFrame containing loaded rows, possibly empty when ``max_rows`` is 0.
    """

    if max_rows is None:
        return pd.read_csv(path)

    chunks = []
    left = max_rows
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if left <= 0:
            break
        piece = chunk.head(left)
        chunks.append(piece)
        left -= len(piece)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _fit_surrogate_from_processed(
    processed_root: Path,
    dataset: str,
    max_interactions: Optional[int] = None,
    n_factors: int = 32,
) -> tuple[LightweightSurrogateRecommender, pd.DataFrame, pd.DataFrame]:
    """Load canonical files, clean core columns, and fit a surrogate recommender.

    Args:
        processed_root: Root directory containing preprocessed datasets.
        dataset: Dataset folder name under ``processed_root``.
        max_interactions: Optional cap on interactions used for fitting.
        n_factors: Number of latent factors for the surrogate model.

    Returns:
        Tuple ``(model, interactions, items)`` after fitting.
    """

    interactions, items = _load_processed_tables(processed_root, dataset, max_interactions=max_interactions)

    model = LightweightSurrogateRecommender(config=SurrogateConfig(n_factors=n_factors))
    model.set_items(items)
    model.fit(interactions)
    return model, interactions, items


def _build_coordinator_and_workers(
    args: argparse.Namespace,
    agent_ids: list[str],
    prompt_store: PromptStore,
    total_steps: int,
) -> tuple[Coordinator, dict[str, object]]:
    """Construct coordinator and worker pool based on CLI settings.

    Args:
        args: Parsed CLI arguments.
        agent_ids: Worker IDs to provision.
        prompt_store: Prompt store instance for policy templates.
        total_steps: Total steps for temperature scheduling.

    Returns:
        Tuple ``(coordinator, workers)``.
    """

    max_snipers = int(getattr(args, "rule_max_snipers", 1))
    if getattr(args, "command", "") == "run-transfer":
        raw_roles = str(getattr(args, "transfer_attack_roles", "all")).strip().lower()
        if raw_roles != "all":
            roles = {r.strip() for r in raw_roles.split(",") if r.strip()}
            if roles.issubset({"sniper", "diagnostic"}) and "sniper" in roles and max_snipers <= 1:
                # Heuristic: boost snipers when transfer keeps only sniper/diagnostic rows.
                max_snipers = max(1, min(3, int(getattr(args, "num_agents", 1))))
        # Keep args in sync for output logging.
        try:
            args.rule_max_snipers = max_snipers
        except Exception:
            pass
    if args.coordinator_policy == "rule":
        policy = RuleBasedCoordinatorPolicy(
            agent_order=agent_ids,
            max_snipers=max_snipers,
        )
    else:
        if args.coordinator_policy == "openai":
            api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI coordinator policy")
            client = build_llm_client(provider="openai", model=args.llm_model, api_key=api_key)
        else:
            client = build_llm_client(provider="ollama", model=args.llm_model, host=args.ollama_host)
        policy = LLMCoordinatorPolicy(
            client=client,
            agent_order=agent_ids,
            prompt_store=prompt_store,
            temperature=args.llm_temperature,
            temperature_end=args.llm_temperature_end,
            total_steps=total_steps,
        )

    lock_role = AgentRole.INACTIVE
    if getattr(args, "sniper_lock_role", None):
        try:
            lock_role = AgentRole(str(args.sniper_lock_role).lower())
        except ValueError:
            lock_role = AgentRole.INACTIVE

    runtime_config = CoordinatorRuntimeConfig(
        profiler_interval=args.profiler_interval,
        profiler_probe_suspicion=args.profiler_probe_suspicion,
        profiler_probe_on_stall=not args.no_profiler_probe_on_stall,
        profiler_probe_suppression_streak=args.profiler_probe_suppression_streak,
        sniper_lock_steps=args.sniper_lock_steps,
        sniper_lock_suspicion=args.sniper_lock_suspicion,
        sniper_lock_suppression_streak=args.sniper_lock_suppression_streak,
        sniper_lock_memory_events=args.sniper_lock_memory_events,
        sniper_lock_role=lock_role,
        transfer_sniper_direct_target=False,
    )
    # If transfer extraction keeps only sniper (and optionally diagnostic) rows,
    # force direct-target snipers so the injected set contains target positives.
    if getattr(args, "command", "") == "run-transfer":
        raw_roles = str(getattr(args, "transfer_attack_roles", "all")).strip().lower()
        if raw_roles != "all":
            roles = {r.strip() for r in raw_roles.split(",") if r.strip()}
            if roles.issubset({"sniper", "diagnostic"}) and "sniper" in roles:
                runtime_config.transfer_sniper_direct_target = True
    coordinator = Coordinator(
        policy=policy,
        runtime_config=runtime_config,
        probe_steps=int(getattr(args, "probe_steps", 2)),
        victim_model_hint=str(getattr(args, "victim_model_hint", "auto")),
        probe_repeats=int(getattr(args, "probe_repeats", 1)),
        probe_use_graph=bool(getattr(args, "probe_use_graph", False)),
        probe_consensus=bool(getattr(args, "probe_consensus", False)),
    )
    worker_policy_name = args.worker_policy
    worker_llm_client = None
    if worker_policy_name != "rule":
        worker_model = args.worker_llm_model or args.llm_model
        if worker_policy_name == "openai":
            api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI worker policy")
            worker_llm_client = build_llm_client(provider="openai", model=worker_model, api_key=api_key)
        else:
            worker_llm_client = build_llm_client(provider="ollama", model=worker_model, host=args.ollama_host)
    _graph_sniper = bool(getattr(args, "graph_sniper", False))
    worker_policy_config = WorkerPolicyConfig(
        implicit_safe=bool(getattr(args, "implicit_safe_attack", False)),
        positive_threshold=float(getattr(args, "target_positive_threshold", 4.0)),
        graph_sniper=_graph_sniper,
        graph_sniper_neighbor_actions=int(getattr(args, "graph_sniper_neighbor_actions", 3)),
        graph_sniper_include_target=bool(getattr(args, "graph_sniper_include_target", True)),
        lightgcn_budget=str(getattr(args, "lightgcn_budget", "small")).lower(),
    )
    workers = build_worker_pool(
        agent_ids,
        policy_config=worker_policy_config,
        llm_client=worker_llm_client,
        prompt_store=prompt_store,
        policy_name=worker_policy_name,
        temperature=args.worker_llm_temperature,
        temperature_end=args.worker_llm_temperature_end,
        total_steps=total_steps,
    )
    return coordinator, workers


def _parse_target_models(raw: str) -> list[str]:
    """Parse comma-separated target model names."""

    out: list[str] = []
    for token in raw.split(","):
        name = token.strip().lower()
        if not name:
            continue
        if name not in {"neumf", "lightgcn", "sequential"}:
            raise ValueError(f"Unsupported target model: {name}")
        out.append(name)
    if not out:
        raise ValueError("At least one target model must be specified")
    return out


def _build_target_config(args: argparse.Namespace) -> TargetModelConfig:
    """Build target model config from CLI args."""

    return TargetModelConfig(
        embedding_dim=args.target_embedding_dim,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        lr=args.target_lr,
        weight_decay=args.target_weight_decay,
        num_negatives=args.target_num_negatives,
        positive_threshold=args.target_positive_threshold,
        implicit_only=args.target_implicit_only,
        explicit_negative_threshold=float(getattr(args, "target_explicit_negative_threshold", 2.0)),
        seed=args.seed,
        device=args.target_device,
        lightgcn_layers=args.target_lightgcn_layers,
    )


def _build_target_model(name: str, config: TargetModelConfig):
    """Instantiate a target recommender by name."""

    if name == "neumf":
        return NeuMFRecommender(config=config)
    if name == "lightgcn":
        return LightGCNRecommender(config=config)
    if name == "sequential":
        return SequentialRecommender(config=config)
    raise ValueError(f"Unsupported target model: {name}")


def _build_episode_recommender(
    name: str,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    *,
    n_factors: int,
    target_config: TargetModelConfig,
):
    """Fit the recommender used during the episode loop for transfer runs."""

    episode_name = str(name).strip().lower()
    if episode_name == "surrogate":
        model = LightweightSurrogateRecommender(config=SurrogateConfig(n_factors=n_factors))
        model.set_items(items)
        model.fit(interactions)
        return model

    model = _build_target_model(episode_name, target_config)
    model.fit(interactions, items=items)
    return model


def _extract_attack_interactions(history: list[dict], roles: set[str] | None = None) -> pd.DataFrame:
    """Extract accepted action outcomes as interaction rows.

    Args:
        history: Episode history produced by ``AGASEpisodeRunner``.
        roles: Optional set of role names (e.g. ``{"sniper"}``) to restrict
            which interactions are extracted. When ``None`` all accepted
            interactions are included regardless of role.

    Returns:
        DataFrame of accepted interactions with effective ratings.
    """

    rows: list[dict] = []
    for entry in history:
        feedback = entry.get("feedback") or {}
        assignments = entry.get("assignments") or {}
        for outcome in feedback.get("outcomes", []):
            if not outcome.get("accepted"):
                continue
            effective_rating = outcome.get("effective_rating")
            if effective_rating is None:
                continue
            action = outcome.get("action", {})
            if roles is not None:
                agent_id = str(action.get("agent_id", ""))
                role = str((assignments.get(agent_id) or {}).get("role", "")).lower()
                if role not in roles:
                    continue
            rows.append(
                {
                    "user_id": str(action.get("agent_id")),
                    "item_id": str(action.get("item_id")),
                    "rating": float(effective_rating),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["user_id", "item_id", "rating"])
    return pd.DataFrame(rows)


def _attack_outcome_stats(history: list[dict], positive_threshold: float) -> dict:
    """Compute basic stats over action outcomes in an episode history.

    This is used for transfer evaluation reports to distinguish:
    - attempted actions (including dropped by defenses),
    - accepted actions,
    - and how many accepted actions are "positive" under the target-model threshold.

    Args:
        history: Per-step episode timeline as returned by ``AGASEpisodeRunner``.
        positive_threshold: Threshold used to define a positive interaction in implicit-feedback target models.

    Returns:
        Dictionary with attempted/accepted counts and a coarse breakdown of accepted effective ratings.
    """

    attempted = 0
    accepted = 0
    accepted_positive = 0
    accepted_nonpositive = 0
    accepted_discounted = 0

    thr = float(positive_threshold)
    for entry in history:
        feedback = entry.get("feedback") or {}
        for outcome in feedback.get("outcomes", []):
            attempted += 1
            if not outcome.get("accepted"):
                continue
            accepted += 1
            eff = outcome.get("effective_rating")
            if eff is None:
                continue
            eff_f = float(eff)
            if eff_f >= thr:
                accepted_positive += 1
            else:
                accepted_nonpositive += 1
            # "Discounted" means accepted but changed in magnitude by the defense.
            action = outcome.get("action") or {}
            raw = action.get("rating")
            if raw is not None:
                try:
                    if abs(float(raw) - eff_f) > 1e-6:
                        accepted_discounted += 1
                except (TypeError, ValueError):
                    pass

    return {
        "attempted_outcomes": int(attempted),
        "accepted_outcomes": int(accepted),
        "accepted_positive_by_threshold": int(accepted_positive),
        "accepted_nonpositive_by_threshold": int(accepted_nonpositive),
        "accepted_discounted": int(accepted_discounted),
        "positive_threshold": float(thr),
    }


def _summarize_goal_status(initial_rank: int, history: list[dict], goal_rank: int) -> dict:
    """Summarize best observed rank and whether the episode reached the target goal.

    Args:
        initial_rank: Target item rank before any AGAS action is executed.
        history: Per-step episode timeline as returned by ``AGASEpisodeRunner``.
        goal_rank: Desired threshold rank, e.g. 5 means "reach rank 5 or better".

    Returns:
        Dictionary containing best-rank and goal-achievement summary fields.
    """

    best_rank = int(initial_rank)
    best_rank_step = None
    goal_first_reached_step = None

    for step_entry in history:
        step = step_entry.get("step")
        rank = step_entry.get("feedback", {}).get("target_rank")
        if not isinstance(rank, int):
            continue
        if rank < best_rank:
            best_rank = int(rank)
            best_rank_step = int(step)
        if goal_first_reached_step is None and rank <= goal_rank:
            goal_first_reached_step = int(step)

    goal_achieved = initial_rank <= goal_rank or goal_first_reached_step is not None
    return {
        "goal_rank": int(goal_rank),
        "best_rank": int(best_rank),
        "best_rank_step": best_rank_step,
        "goal_achieved": bool(goal_achieved),
        "goal_first_reached_step": goal_first_reached_step,
    }


def cmd_preprocess(args: argparse.Namespace) -> int:
    """CLI handler for preprocessing raw datasets into canonical CSV outputs.

    Args:
        args: Parsed command-line arguments for the ``preprocess`` subcommand.

    Returns:
        Process exit code.
    """

    if preprocess_all is None:
        print("ERROR: agas.data module is not installed. Cannot run preprocess command.")
        return 1

    include = None
    if args.datasets:
        include = {d.strip() for d in args.datasets.split(",") if d.strip()}

    cfg = PreprocessConfig(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        chunk_size=args.chunk_size,
        max_rows_per_dataset=_parse_optional_int(args.max_rows_per_dataset),
        overwrite=not args.no_overwrite,
        include_datasets=include,
    )

    stats = preprocess_all(cfg)
    print("Preprocessing complete.")
    for row in stats:
        print(
            f"- {row.dataset}: interactions={row.interactions}, "
            f"unique_users={row.unique_users}, unique_items={row.unique_items}"
        )
    print(f"Summary written to {cfg.output_root / 'summary.csv'}")
    return 0


def cmd_train_surrogate(args: argparse.Namespace) -> int:
    """CLI handler to fit and serialize a surrogate model pickle.

    Args:
        args: Parsed command-line arguments for ``train-surrogate``.

    Returns:
        Process exit code.
    """

    model, interactions, items = _fit_surrogate_from_processed(
        processed_root=Path(args.processed_root),
        dataset=args.dataset,
        max_interactions=_parse_optional_int(args.max_interactions),
        n_factors=args.n_factors,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"model": model, "interactions": interactions, "items": items}, f)

    print(f"Surrogate model saved to {output_path}")
    print(
        f"Interactions used: {len(interactions)} | "
        f"Users: {interactions['user_id'].nunique()} | "
        f"Items: {interactions['item_id'].nunique()}"
    )
    return 0


def cmd_run_episode(args: argparse.Namespace) -> int:
    """CLI handler to run one AGAS episode and persist a full JSON trace.

    Args:
        args: Parsed command-line arguments for ``run-episode``.

    Returns:
        Process exit code.
    """

    model, interactions, items = _fit_surrogate_from_processed(
        processed_root=Path(args.processed_root),
        dataset=args.dataset,
        max_interactions=_parse_optional_int(args.max_interactions),
        n_factors=args.n_factors,
    )
    target_item_id = str(args.target_item_id)

    agent_ids = default_agent_ids(args.num_agents)
    prompt_store = PromptStore(Path(args.prompt_root))
    coordinator, workers = _build_coordinator_and_workers(
        args=args,
        agent_ids=agent_ids,
        prompt_store=prompt_store,
        total_steps=args.num_steps,
    )

    env = AGASEnvironment(
        recommender=model,
        base_interactions=interactions,
        items=items,
        target_item_id=target_item_id,
        target_keyword=args.target_keyword,
        defense_config=DefenseConfig(
            black_box_mode=not args.expose_defense_state,
            spike_threshold=args.spike_threshold,
            lockdown_drop_prob=args.lockdown_drop_prob,
            monitor_config=DefenseMonitorConfig(
                group_overlap_threshold=args.group_overlap_threshold,
                group_target_required=args.group_target_required,
                group_weight=args.group_weight,
            ),
            quarantine_steps=int(args.quarantine_steps),
            quarantine_on_spike_alert=bool(args.quarantine_on_spike_alert),
            quarantine_on_group_collusion=bool(args.quarantine_on_group_collusion),
        ),
        seed=args.seed,
    )
    initial_rank = int(env.current_rank)
    initial_total_candidates = int(env.total_candidates)
    print(
        f"Initial target rank for item {target_item_id}: "
        f"{initial_rank}/{initial_total_candidates} | goal <= {args.goal_rank}"
    )

    runner = AGASEpisodeRunner(
        coordinator=coordinator,
        environment=env,
        workers=workers,
        config=EpisodeConfig(
            num_steps=args.num_steps,
            num_workers=args.num_agents,
            goal_rank=args.goal_rank,
            stop_on_goal=args.stop_on_goal,
            trajectory_window=args.trajectory_window,
            coordinator_agent_memory=bool(args.coordinator_agent_memory),
        ),
    )
    result = runner.run()
    goal_summary = _summarize_goal_status(initial_rank=initial_rank, history=result.history, goal_rank=args.goal_rank)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "target_item_id": target_item_id,
                "target_keyword": args.target_keyword,
                "num_steps": args.num_steps,
                "num_agents": args.num_agents,
                "coordinator_policy": args.coordinator_policy,
                "worker_policy": args.worker_policy,
                "prompt_root": str(Path(args.prompt_root)),
                "initial_rank": initial_rank,
                "initial_total_candidates": initial_total_candidates,
                "stop_on_goal": bool(args.stop_on_goal),
                "stopped_early": bool(result.stopped_early),
                "stop_reason": result.stop_reason,
                "executed_steps": result.executed_steps,
                "goal_rank": goal_summary["goal_rank"],
                "best_rank": goal_summary["best_rank"],
                "best_rank_step": goal_summary["best_rank_step"],
                "goal_achieved": goal_summary["goal_achieved"],
                "goal_first_reached_step": goal_summary["goal_first_reached_step"],
                "final_rank": result.final_rank,
                "final_total_candidates": result.final_total_candidates,
                "final_worker_states": result.final_worker_states,
                "agent_logs": result.agent_logs,
                "coordinator_logs": result.coordinator_logs,
                "history": result.history,
            },
            f,
            indent=2,
        )

    print(f"Episode finished. Target rank: {result.final_rank}/{result.final_total_candidates}")
    if result.stopped_early:
        print(f"Stopped early: yes ({result.stop_reason})")
    else:
        print("Stopped early: no")
    best_rank_step = goal_summary["best_rank_step"]
    best_rank_step_label = "initial" if best_rank_step is None else f"step {best_rank_step}"
    print(f"Best target rank observed: {goal_summary['best_rank']} ({best_rank_step_label})")
    if goal_summary["goal_achieved"]:
        goal_step = goal_summary["goal_first_reached_step"]
        goal_step_label = "before step 0" if goal_step is None else f"step {goal_step}"
        print(f"Goal achieved: yes (target rank <= {args.goal_rank} reached at {goal_step_label})")
    else:
        print(f"Goal achieved: no (target rank <= {args.goal_rank} was not reached)")
    for step_entry in result.history:
        step = step_entry["step"]
        feedback = step_entry["feedback"]
        rank = feedback["target_rank"]
        total = feedback["total_candidates"]
        print(f"Step {step}: target rank {rank}/{total}")
        for report in step_entry["reports"]:
            actions = report.get("actions", [])
            action_text = ", ".join(f"{a['item_id']}->{a['rating']}" for a in actions) if actions else "no action"
            print(f"  {report['agent_id']} [{report['role']}] via {report.get('policy', 'rule')}: {action_text}")
    print(f"Saved detailed timeline to {out_path}")
    return 0


def cmd_run_transfer(args: argparse.Namespace) -> int:
    """Run transfer evaluation for Option A (offline) and/or Option B (in-loop).

    Args:
        args: Parsed command-line arguments for ``run-transfer``.

    Returns:
        Process exit code.
    """

    target_item_id = str(args.target_item_id)
    mode = args.transfer_mode
    target_models = _parse_target_models(args.target_models)
    target_config = _build_target_config(args)

    interactions, items = _load_processed_tables(
        processed_root=Path(args.processed_root),
        dataset=args.dataset,
        max_interactions=_parse_optional_int(args.max_interactions),
    )

    # Many evaluation protocols assume a fixed item catalog. If the target item is not in the
    # clean catalog, "promotion" deltas are dominated by catalog inclusion (cold-start) artifacts.
    catalog_item_ids = set(items["item_id"].astype(str).drop_duplicates().tolist()) if items is not None else set()
    if not args.allow_cold_start_target and target_item_id not in catalog_item_ids:
        raise RuntimeError(
            f"Target item {target_item_id} is not present in processed items.csv for dataset={args.dataset}. "
            "Pick a target that exists in the clean catalog, or pass --allow-cold-start-target to proceed "
            "(treating this as a cold-start/inclusion setting rather than pure rank promotion)."
        )

    prompt_store = PromptStore(Path(args.prompt_root))

    # RC4 fix: restrict which roles contribute interactions to target model training.
    _raw_roles = str(getattr(args, "transfer_attack_roles", "all")).strip().lower()
    attack_roles: set[str] | None = None if _raw_roles == "all" else {r.strip() for r in _raw_roles.split(",")}

    def _build_env_and_candidate(episode_model_name: str, candidate_set: str):
        episode_recommender = _build_episode_recommender(
            episode_model_name,
            interactions,
            items,
            n_factors=args.n_factors,
            target_config=target_config,
        )
        env = AGASEnvironment(
            recommender=episode_recommender,
            base_interactions=interactions,
            items=items,
            target_item_id=target_item_id,
            target_keyword=args.target_keyword,
            defense_config=DefenseConfig(
                black_box_mode=not args.expose_defense_state,
                spike_threshold=args.spike_threshold,
                lockdown_drop_prob=args.lockdown_drop_prob,
                monitor_config=DefenseMonitorConfig(
                    group_overlap_threshold=args.group_overlap_threshold,
                    group_target_required=args.group_target_required,
                    group_weight=args.group_weight,
                ),
                quarantine_steps=int(args.quarantine_steps),
                quarantine_on_spike_alert=bool(args.quarantine_on_spike_alert),
                quarantine_on_group_collusion=bool(args.quarantine_on_group_collusion),
            ),
            seed=args.seed,
        )
        if candidate_set == "all_items":
            candidate_items = items["item_id"].astype(str).drop_duplicates().tolist()
        else:
            candidate_items = list(env.target_cluster_item_ids)
        if target_item_id not in candidate_items:
            candidate_items.append(target_item_id)
        if candidate_set == "all_items" and len(candidate_items) > 5000:
            print(
                f"NOTE: 'all_items' candidate set has {len(candidate_items)} items. "
                "Absolute rank numbers will be large; compare using normalized_rank_delta in the output."
            )
        return env, candidate_items, env.segment_user_ids

    split_by_target = bool(getattr(args, "transfer_split_by_target_model", False))

    surrogate_result = None
    attack_rows = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    attack_stats = None
    episode_histories_by_target: dict[str, list[dict]] | None = None
    per_target_settings: dict[str, dict] | None = None
    if mode in {"both", "option-a"} and not split_by_target:
        episode_model = str(getattr(args, "episode_model", "surrogate")).strip().lower()
        base_env, candidate_items, segment_user_ids = _build_env_and_candidate(
            episode_model, args.transfer_candidate_set
        )

        # RC1+RC2 fix: reuse real segment users as attack agents so they already have
        # embeddings and graph edges in the target models trained on clean data.
        if getattr(args, "use_segment_users_as_agents", False):
            if len(segment_user_ids) < args.num_agents:
                raise RuntimeError(
                    f"Not enough segment users ({len(segment_user_ids)}) to fill "
                    f"{args.num_agents} agent slots. Lower --num-agents or disable "
                    "--use-segment-users-as-agents."
                )
            agent_ids = list(segment_user_ids[:args.num_agents])
        else:
            agent_ids = default_agent_ids(args.num_agents)

        coordinator, workers = _build_coordinator_and_workers(
            args=args,
            agent_ids=agent_ids,
            prompt_store=prompt_store,
            total_steps=args.num_steps,
        )
        runner = AGASEpisodeRunner(
            coordinator=coordinator,
            environment=base_env,
            workers=workers,
            config=EpisodeConfig(
                num_steps=args.num_steps,
                num_workers=args.num_agents,
                goal_rank=args.goal_rank,
                stop_on_goal=args.stop_on_goal,
                trajectory_window=args.trajectory_window,
                coordinator_agent_memory=bool(args.coordinator_agent_memory),
            ),
        )
        surrogate_result = runner.run()
        attack_rows = _extract_attack_interactions(surrogate_result.history, roles=attack_roles)
        attack_stats = _attack_outcome_stats(surrogate_result.history, positive_threshold=target_config.positive_threshold)

    option_a_results: dict[str, dict] = {}
    if mode in {"both", "option-a"}:
        if not split_by_target:
            if len(attack_rows):
                combined = pd.concat([interactions, attack_rows], ignore_index=True)
                # RC3 fix: deduplicate (user_id, item_id) pairs keeping the attack rating,
                # which matters when segment users are used as agents and already rated
                # some items in the clean data.
                combined = combined.drop_duplicates(subset=["user_id", "item_id"], keep="last")
            else:
                combined = interactions
            for name in target_models:
                clean_model = _build_target_model(name, target_config)
                clean_model.fit(interactions, items=items)
                initial_rank, initial_total = clean_model.rank_item(
                    item_id=target_item_id,
                    segment_user_ids=segment_user_ids,
                    candidate_items=candidate_items,
                )
                attacked_model = _build_target_model(name, target_config)
                attacked_model.fit(combined, items=items)
                final_rank, final_total = attacked_model.rank_item(
                    item_id=target_item_id,
                    segment_user_ids=segment_user_ids,
                    candidate_items=candidate_items,
                )
                option_a_results[name] = {
                    "initial_rank": int(initial_rank),
                    "initial_total_candidates": int(initial_total),
                    "final_rank": int(final_rank),
                    "final_total_candidates": int(final_total),
                    "rank_delta": int(initial_rank) - int(final_rank),
                    "normalized_rank_before": round(initial_rank / initial_total, 4) if initial_total else None,
                    "normalized_rank_after": round(final_rank / final_total, 4) if final_total else None,
                    "normalized_rank_delta": round((initial_rank - final_rank) / initial_total, 4) if initial_total else None,
                    "attack_interactions": int(len(attack_rows)),
                }
        else:
            episode_histories_by_target = {}
            per_target_settings = {}
            for name in target_models:
                # Per-target overrides: LightGCN gets real users + direct target edges + more snipers.
                use_segment_users = bool(getattr(args, "use_segment_users_as_agents", False))
                episode_model = str(getattr(args, "episode_model", "surrogate")).strip().lower()
                victim_hint = str(getattr(args, "victim_model_hint", "auto")).strip().lower()
                probe_steps = int(getattr(args, "probe_steps", 2))
                rule_max_snipers = int(getattr(args, "rule_max_snipers", 1))
                candidate_set = str(getattr(args, "transfer_candidate_set", "cluster"))

                if name == "lightgcn":
                    use_segment_users = True
                    episode_model = "lightgcn"
                    victim_hint = "mf"
                    probe_steps = 0
                    rule_max_snipers = max(rule_max_snipers, 3)
                    candidate_set = "all_items"

                per_target_settings[name] = {
                    "episode_model": episode_model,
                    "victim_model_hint": victim_hint,
                    "probe_steps": probe_steps,
                    "use_segment_users_as_agents": use_segment_users,
                    "rule_max_snipers": rule_max_snipers,
                    "transfer_candidate_set": candidate_set,
                }

                base_env, candidate_items, segment_user_ids = _build_env_and_candidate(
                    episode_model, candidate_set
                )

                if use_segment_users:
                    if len(segment_user_ids) < args.num_agents:
                        raise RuntimeError(
                            f"Not enough segment users ({len(segment_user_ids)}) to fill "
                            f"{args.num_agents} agent slots for target model {name}. "
                            "Lower --num-agents or disable --transfer-split-by-target-model."
                        )
                    agent_ids = list(segment_user_ids[:args.num_agents])
                else:
                    agent_ids = default_agent_ids(args.num_agents)

                # Clone args for per-target overrides
                args_local = argparse.Namespace(**vars(args))
                args_local.episode_model = episode_model
                args_local.victim_model_hint = victim_hint
                args_local.probe_steps = probe_steps
                args_local.use_segment_users_as_agents = use_segment_users
                args_local.rule_max_snipers = rule_max_snipers
                args_local.transfer_candidate_set = candidate_set

                coordinator, workers = _build_coordinator_and_workers(
                    args=args_local,
                    agent_ids=agent_ids,
                    prompt_store=prompt_store,
                    total_steps=args.num_steps,
                )
                runner = AGASEpisodeRunner(
                    coordinator=coordinator,
                    environment=base_env,
                    workers=workers,
                    config=EpisodeConfig(
                        num_steps=args.num_steps,
                        num_workers=args.num_agents,
                        goal_rank=args.goal_rank,
                        stop_on_goal=args.stop_on_goal,
                        trajectory_window=args.trajectory_window,
                        coordinator_agent_memory=bool(args.coordinator_agent_memory),
                    ),
                )
                result = runner.run()
                episode_histories_by_target[name] = result.history
                attack_rows_local = _extract_attack_interactions(result.history, roles=attack_roles)
                if len(attack_rows_local):
                    combined = pd.concat([interactions, attack_rows_local], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["user_id", "item_id"], keep="last")
                else:
                    combined = interactions

                clean_model = _build_target_model(name, target_config)
                clean_model.fit(interactions, items=items)
                initial_rank, initial_total = clean_model.rank_item(
                    item_id=target_item_id,
                    segment_user_ids=segment_user_ids,
                    candidate_items=candidate_items,
                )
                attacked_model = _build_target_model(name, target_config)
                attacked_model.fit(combined, items=items)
                final_rank, final_total = attacked_model.rank_item(
                    item_id=target_item_id,
                    segment_user_ids=segment_user_ids,
                    candidate_items=candidate_items,
                )
                option_a_results[name] = {
                    "initial_rank": int(initial_rank),
                    "initial_total_candidates": int(initial_total),
                    "final_rank": int(final_rank),
                    "final_total_candidates": int(final_total),
                    "rank_delta": int(initial_rank) - int(final_rank),
                    "normalized_rank_before": round(initial_rank / initial_total, 4) if initial_total else None,
                    "normalized_rank_after": round(final_rank / final_total, 4) if final_total else None,
                    "normalized_rank_delta": round((initial_rank - final_rank) / initial_total, 4) if initial_total else None,
                    "attack_interactions": int(len(attack_rows_local)),
                }

    option_b_results: dict[str, dict] = {}
    if mode in {"both", "option-b"}:
        for name in target_models:
            target_model = _build_target_model(name, target_config)
            target_model.fit(interactions, items=items)
            env = AGASEnvironment(
                recommender=target_model,
                base_interactions=interactions,
                items=items,
                target_item_id=target_item_id,
                target_keyword=args.target_keyword,
                defense_config=DefenseConfig(
                    black_box_mode=not args.expose_defense_state,
                    spike_threshold=args.spike_threshold,
                    lockdown_drop_prob=args.lockdown_drop_prob,
                    monitor_config=DefenseMonitorConfig(
                        group_overlap_threshold=args.group_overlap_threshold,
                        group_target_required=args.group_target_required,
                        group_weight=args.group_weight,
                    ),
                    quarantine_steps=int(args.quarantine_steps),
                    quarantine_on_spike_alert=bool(args.quarantine_on_spike_alert),
                    quarantine_on_group_collusion=bool(args.quarantine_on_group_collusion),
                ),
                seed=args.seed,
            )
            initial_rank = int(env.current_rank)
            coordinator, workers = _build_coordinator_and_workers(
                args=args,
                agent_ids=agent_ids,
                prompt_store=prompt_store,
                total_steps=args.num_steps,
            )
            runner = AGASEpisodeRunner(
                coordinator=coordinator,
                environment=env,
                workers=workers,
                config=EpisodeConfig(
                    num_steps=args.num_steps,
                    num_workers=args.num_agents,
                    goal_rank=args.goal_rank,
                    stop_on_goal=args.stop_on_goal,
                    trajectory_window=args.trajectory_window,
                    coordinator_agent_memory=bool(args.coordinator_agent_memory),
                ),
            )
            result = runner.run()
            goal_summary = _summarize_goal_status(initial_rank=initial_rank, history=result.history, goal_rank=args.goal_rank)
            option_b_results[name] = {
                "initial_rank": int(initial_rank),
                "initial_total_candidates": int(env.total_candidates),
                "final_rank": int(result.final_rank),
                "final_total_candidates": int(result.final_total_candidates),
                "best_rank": int(goal_summary["best_rank"]),
                "best_rank_step": goal_summary["best_rank_step"],
                "goal_achieved": bool(goal_summary["goal_achieved"]),
                "goal_first_reached_step": goal_summary["goal_first_reached_step"],
                "stopped_early": bool(result.stopped_early),
                "stop_reason": result.stop_reason,
                "history": result.history,
            }

    output = {
        "dataset": args.dataset,
        "processed_root": str(Path(args.processed_root)),
        "max_interactions": str(args.max_interactions),
        "n_factors": int(args.n_factors),
        "seed": int(args.seed),
        "episode_model": str(episode_model),
        "implicit_safe_attack": bool(getattr(args, "implicit_safe_attack", False)),
        "graph_sniper": bool(getattr(args, "graph_sniper", False)),
        "victim_model_hint": str(getattr(args, "victim_model_hint", "auto")),
        "probe_steps": int(getattr(args, "probe_steps", 2)),
        "lightgcn_budget": str(getattr(args, "lightgcn_budget", "small")),
        "rule_max_snipers": int(getattr(args, "rule_max_snipers", 1)),
        "target_item_id": target_item_id,
        "target_keyword": args.target_keyword,
        "num_steps": args.num_steps,
        "num_agents": args.num_agents,
        "coordinator_policy": args.coordinator_policy,
        "worker_policy": args.worker_policy,
        "transfer_mode": mode,
        "target_models": target_models,
        "transfer_candidate_set": str(args.transfer_candidate_set),
        "transfer_candidate_set_size": int(len(candidate_items)),
        "target_embedding_dim": int(target_config.embedding_dim),
        "target_epochs": int(target_config.epochs),
        "target_batch_size": int(target_config.batch_size),
        "target_lr": float(target_config.lr),
        "target_weight_decay": float(target_config.weight_decay),
        "target_num_negatives": int(target_config.num_negatives),
        "target_positive_threshold": float(target_config.positive_threshold),
        "target_implicit_only": bool(target_config.implicit_only),
        "target_device": str(target_config.device),
        "target_lightgcn_layers": int(target_config.lightgcn_layers),
        "use_segment_users_as_agents": bool(getattr(args, "use_segment_users_as_agents", False)),
        "transfer_attack_roles": str(getattr(args, "transfer_attack_roles", "all")),
        "option_a": option_a_results if option_a_results else None,
        "option_b": option_b_results if option_b_results else None,
        "surrogate_episode_history": surrogate_result.history if surrogate_result is not None else None,
        "episode_histories_by_target": episode_histories_by_target,
        "per_target_transfer_settings": per_target_settings,
        "attack_interactions_stats": attack_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Transfer evaluation saved to {out_path}")
    if option_a_results:
        print("Option A (offline target models):")
        for name, stats in option_a_results.items():
            print(
                f"  {name}: rank {stats['initial_rank']}->{stats['final_rank']} "
                f"(delta {stats['rank_delta']}) using {stats['attack_interactions']} attack interactions"
            )
    if option_b_results:
        print("Option B (in-loop target models):")
        for name, stats in option_b_results.items():
            print(f"  {name}: final rank {stats['final_rank']} (best {stats['best_rank']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``agas`` parser and all subcommands."""

    parser = argparse.ArgumentParser(prog="agas", description="AGAS framework CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Preprocess raw datasets under data/")
    p_pre.add_argument("--data-root", default="data")
    p_pre.add_argument("--output-root", default="processed")
    p_pre.add_argument("--chunk-size", type=int, default=200_000)
    p_pre.add_argument(
        "--max-rows-per-dataset",
        default="2000000",
        help="Rows per dataset to export. Use 'none' for full export.",
    )
    p_pre.add_argument("--datasets", default=None, help="Comma-separated dataset names to preprocess")
    p_pre.add_argument("--no-overwrite", action="store_true")
    p_pre.set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train-surrogate", help="Fit and persist surrogate model")
    p_train.add_argument("--processed-root", default="processed")
    p_train.add_argument("--dataset", default="ml-latest-small")
    p_train.add_argument("--max-interactions", default="500000")
    p_train.add_argument("--n-factors", type=int, default=32)
    p_train.add_argument("--output", default="outputs/surrogate.pkl")
    p_train.set_defaults(func=cmd_train_surrogate)

    p_run = sub.add_parser("run-episode", help="Run AGAS episode simulation")
    p_run.add_argument("--processed-root", default="processed")
    p_run.add_argument("--dataset", default="ml-latest-small")
    p_run.add_argument("--max-interactions", default="500000")
    p_run.add_argument("--n-factors", type=int, default=32)
    p_run.add_argument("--target-item-id", default="101")
    p_run.add_argument("--target-keyword", default="horror")
    p_run.add_argument("--num-agents", type=int, default=4)
    p_run.add_argument("--num-steps", type=int, default=4)
    p_run.add_argument("--goal-rank", type=int, default=5)
    p_run.add_argument(
        "--stop-on-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop the episode early once the target rank goal is reached.",
    )
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--trajectory-window", type=int, default=5)
    p_run.add_argument(
        "--coordinator-agent-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-agent recent outcomes in coordinator observation.",
    )

    p_run.add_argument("--coordinator-policy", choices=["rule", "openai", "ollama"], default="openai")
    p_run.add_argument("--llm-model", default="gpt-5-mini")
    p_run.add_argument("--llm-temperature", type=float, default=0.3)
    p_run.add_argument("--llm-temperature-end", type=float, default=None)
    p_run.add_argument("--openai-api-key", default=None)
    p_run.add_argument("--ollama-host", default="http://localhost:11434")
    p_run.add_argument("--worker-policy", choices=["rule", "openai", "ollama"], default="rule")
    p_run.add_argument("--worker-llm-model", default=None)
    p_run.add_argument("--worker-llm-temperature", type=float, default=0.3)
    p_run.add_argument("--worker-llm-temperature-end", type=float, default=None)
    p_run.add_argument("--prompt-root", default="prompts")
    p_run.add_argument("--expose-defense-state", action="store_true")

    p_run.add_argument("--spike-threshold", type=int, default=2)
    p_run.add_argument("--lockdown-drop-prob", type=float, default=0.55)
    p_run.add_argument("--quarantine-steps", type=int, default=0)
    p_run.add_argument(
        "--quarantine-on-spike-alert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quarantine accounts after spike detector triggers.",
    )
    p_run.add_argument(
        "--quarantine-on-group-collusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quarantine accounts after group-collusion triggers.",
    )
    p_run.add_argument("--profiler-interval", type=int, default=3)
    p_run.add_argument("--profiler-probe-suspicion", type=float, default=0.35)
    p_run.add_argument(
        "--no-profiler-probe-on-stall",
        action="store_true",
        help="Disable profiler probes when target rank stalls.",
    )
    p_run.add_argument("--profiler-probe-suppression-streak", type=int, default=1)
    p_run.add_argument("--sniper-lock-steps", type=int, default=2)
    p_run.add_argument("--sniper-lock-suspicion", type=float, default=0.6)
    p_run.add_argument("--sniper-lock-suppression-streak", type=int, default=2)
    p_run.add_argument(
        "--sniper-lock-memory-events",
        type=int,
        default=2,
        help="Lock a sniper if its recent per-agent memory contains at least this many drop/discount events.",
    )
    p_run.add_argument(
        "--sniper-lock-role",
        choices=["inactive", "camouflaguer"],
        default="inactive",
        help="Role to assign when a sniper is locked after suppression.",
    )
    p_run.add_argument(
        "--rule-max-snipers",
        type=int,
        default=1,
        help="For rule-based coordinator: maximum number of snipers to assign per step when target rank is still > 5.",
    )
    p_run.add_argument("--group-overlap-threshold", type=float, default=0.6)
    p_run.add_argument("--group-target-required", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--group-weight", type=float, default=0.2)

    p_run.add_argument("--output", default="outputs/episode_result.json")
    p_run.set_defaults(func=cmd_run_episode)

    p_transfer = sub.add_parser("run-transfer", help="Run transfer evaluation on target models")
    p_transfer.add_argument("--processed-root", default="processed")
    p_transfer.add_argument("--dataset", default="ml-latest-small")
    p_transfer.add_argument("--max-interactions", default="500000")
    p_transfer.add_argument("--n-factors", type=int, default=32)
    p_transfer.add_argument("--target-item-id", default="101")
    p_transfer.add_argument("--target-keyword", default="horror")
    p_transfer.add_argument("--num-agents", type=int, default=4)
    p_transfer.add_argument("--num-steps", type=int, default=4)
    p_transfer.add_argument("--goal-rank", type=int, default=5)
    p_transfer.add_argument(
        "--stop-on-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the episode early once the target rank goal is reached.",
    )
    p_transfer.add_argument("--seed", type=int, default=42)
    p_transfer.add_argument("--trajectory-window", type=int, default=5)
    p_transfer.add_argument(
        "--coordinator-agent-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-agent recent outcomes in coordinator observation.",
    )

    p_transfer.add_argument("--coordinator-policy", choices=["rule", "openai", "ollama"], default="openai")
    p_transfer.add_argument("--llm-model", default="gpt-5-mini")
    p_transfer.add_argument("--llm-temperature", type=float, default=0.3)
    p_transfer.add_argument("--llm-temperature-end", type=float, default=None)
    p_transfer.add_argument("--openai-api-key", default=None)
    p_transfer.add_argument("--ollama-host", default="http://localhost:11434")
    p_transfer.add_argument("--worker-policy", choices=["rule", "openai", "ollama"], default="rule")
    p_transfer.add_argument("--worker-llm-model", default=None)
    p_transfer.add_argument("--worker-llm-temperature", type=float, default=0.3)
    p_transfer.add_argument("--worker-llm-temperature-end", type=float, default=None)
    p_transfer.add_argument("--prompt-root", default="prompts")
    p_transfer.add_argument("--expose-defense-state", action="store_true")

    p_transfer.add_argument("--spike-threshold", type=int, default=2)
    p_transfer.add_argument("--lockdown-drop-prob", type=float, default=0.55)
    p_transfer.add_argument("--quarantine-steps", type=int, default=0)
    p_transfer.add_argument(
        "--quarantine-on-spike-alert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quarantine accounts after spike detector triggers.",
    )
    p_transfer.add_argument(
        "--quarantine-on-group-collusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quarantine accounts after group-collusion triggers.",
    )
    p_transfer.add_argument("--profiler-interval", type=int, default=3)
    p_transfer.add_argument("--profiler-probe-suspicion", type=float, default=0.35)
    p_transfer.add_argument(
        "--no-profiler-probe-on-stall",
        action="store_true",
        help="Disable profiler probes when target rank stalls.",
    )
    p_transfer.add_argument("--profiler-probe-suppression-streak", type=int, default=1)
    p_transfer.add_argument("--sniper-lock-steps", type=int, default=2)
    p_transfer.add_argument("--sniper-lock-suspicion", type=float, default=0.6)
    p_transfer.add_argument("--sniper-lock-suppression-streak", type=int, default=2)
    p_transfer.add_argument(
        "--sniper-lock-memory-events",
        type=int,
        default=2,
        help="Lock a sniper if its recent per-agent memory contains at least this many drop/discount events.",
    )
    p_transfer.add_argument(
        "--sniper-lock-role",
        choices=["inactive", "camouflaguer"],
        default="inactive",
        help="Role to assign when a sniper is locked after suppression.",
    )
    p_transfer.add_argument("--group-overlap-threshold", type=float, default=0.6)
    p_transfer.add_argument("--group-target-required", action=argparse.BooleanOptionalAction, default=True)
    p_transfer.add_argument("--group-weight", type=float, default=0.2)

    p_transfer.add_argument("--transfer-mode", choices=["both", "option-a", "option-b"], default="both")
    p_transfer.add_argument(
        "--implicit-safe-attack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, the episode-loop workers avoid emitting ratings >= --target-positive-threshold "
            "for profiler/camouflaguer roles (to avoid creating unintended implicit positives during offline transfer). "
            "Sniper keeps pushing the target with 5.0 and skips competitor downrating for implicit-only training."
        ),
    )
    p_transfer.add_argument(
        "--graph-sniper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable graph-aware sniper mode for degree-normalised models such as LightGCN. "
            "Instead of rating the target item directly (which inflates its degree and dilutes "
            "existing edges via D^{-1/2} A D^{-1/2}), snipers rate cluster-neighbour items at "
            "5.0 (graph diffusion lifts the target) and competitors at 5.0 (inflates competitor "
            "degrees, weakening their edges). The target item is never rated directly."
        ),
    )
    p_transfer.add_argument(
        "--graph-sniper-neighbor-actions",
        type=int,
        default=3,
        help="Number of cluster-neighbour items each sniper rates in --graph-sniper mode (default: 3).",
    )
    p_transfer.add_argument(
        "--graph-sniper-include-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When graph-sniper mode is active, append a direct 5.0 rating on the target item as "
            "the last sniper action. Default True because offline transfer attacks retrain the "
            "victim from scratch — degree normalisation adapts, so a direct target positive is "
            "beneficial and ensures --transfer-attack-roles sniper produces at least one target "
            "positive. Set --no-graph-sniper-include-target only for pure online-injection attacks "
            "where the LightGCN model is never retrained."
        ),
    )
    p_transfer.add_argument(
        "--victim-model-hint",
        choices=["auto", "mf", "lightgcn", "sequential"],
        default="auto",
        help=(
            "Victim recommender architecture hint. 'auto' (default) runs a probe phase for "
            "--probe-steps steps to classify the victim and then adapts the attack strategy. "
            "'mf' forces MF/NeuMF-style strategy (direct 5.0 target rating, sparse fake profiles). "
            "'lightgcn' forces graph-sniper strategy (no direct target rating; cluster-neighbour "
            "and competitor degree inflation). "
            "'sequential' forces sequential-sniper strategy (filler items first, target last)."
        ),
    )
    p_transfer.add_argument(
        "--probe-steps",
        type=int,
        default=2,
        help=(
            "Number of steps dedicated to probing the victim model architecture when "
            "--victim-model-hint auto is set. Step 0 tests direct target rating (MF vs LightGCN). "
            "Step 1 tests recency ordering (MF vs Sequential). Set to 0 to skip probing entirely "
            "and rely on --victim-model-hint. (default: 2)"
        ),
    )
    p_transfer.add_argument(
        "--probe-repeats",
        type=int,
        default=1,
        help="Repeat each probe type this many times and use the average delta (default: 1).",
    )
    p_transfer.add_argument(
        "--probe-use-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable graph-diffusion probe (cluster-neighbour ratings) to detect LightGCN-style victims.",
    )
    p_transfer.add_argument(
        "--probe-consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require majority-consensus over repeated probes before classifying the victim model.",
    )
    p_transfer.add_argument(
        "--lightgcn-budget",
        choices=["small", "large"],
        default="small",
        help=(
            "Interaction budget mode for LightGCN-style victims. "
            "'small' (default) uses fake users (cold-start): profiler builds graph proximity by "
            "rating cluster neighbours before snipers fire; camouflageur creates diffusion pathways. "
            "'large' uses real users with established graph connections: profiling is lightweight, "
            "snipers coordinate to cover different cluster neighbors for broad diffusion coverage."
        ),
    )
    p_transfer.add_argument(
        "--episode-model",
        choices=["surrogate", "lightgcn", "neumf", "sequential"],
        default="surrogate",
        help=(
            "Recommender used during the episode loop that generates attack interactions for Option A. "
            "Default 'surrogate' uses the lightweight SVD-based surrogate. "
            "Set to 'lightgcn', 'neumf', or 'sequential' to run the episode loop on that target-model family, then "
            "still evaluate offline transfer to --target-models afterward."
        ),
    )
    p_transfer.add_argument(
        "--transfer-candidate-set",
        choices=["cluster", "all_items"],
        default="cluster",
        help=(
            "Candidate set used when reporting ranks in transfer evaluation. "
            "'cluster' uses the surrogate-derived target cluster (can drift between runs). "
            "'all_items' uses all item IDs from processed items.csv (more stable)."
        ),
    )
    p_transfer.add_argument(
        "--transfer-split-by-target-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run separate episode-generation passes per target model with per-model attack recipes. "
            "LightGCN uses real users + direct target edges (MF-style) + more snipers + all_items "
            "candidate set; NeuMF uses the standard settings. Outputs per-target histories and settings."
        ),
    )
    p_transfer.add_argument(
        "--allow-cold-start-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow transfer evaluation when the target item is not present in the clean item catalog. "
            "This turns the experiment into a cold-start/catalog-inclusion setting where rank deltas "
            "are not directly comparable to 'promote an existing item' scenarios."
        ),
    )
    p_transfer.add_argument("--target-models", default="neumf,lightgcn")
    p_transfer.add_argument("--target-embedding-dim", type=int, default=32)
    p_transfer.add_argument("--target-epochs", type=int, default=3)
    p_transfer.add_argument("--target-batch-size", type=int, default=1024)
    p_transfer.add_argument("--target-lr", type=float, default=1e-3)
    p_transfer.add_argument("--target-weight-decay", type=float, default=1e-5)
    p_transfer.add_argument("--target-num-negatives", type=int, default=4)
    p_transfer.add_argument("--target-positive-threshold", type=float, default=4.0)
    p_transfer.add_argument(
        "--target-explicit-negative-threshold",
        type=float,
        default=2.0,
        help=(
            "In implicit-only mode, interactions with rating <= this threshold are used as "
            "explicit BPR negatives instead of randomly sampled ones. This gives competitor "
            "downratings (rating=1.0 from sniper) real training signal. Set to 0 to disable."
        ),
    )
    p_transfer.add_argument(
        "--target-implicit-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Train target models as implicit-feedback recommenders: only rows with rating >= "
            "--target-positive-threshold are treated as observed (positive) interactions. "
            "Disable (set --no-target-implicit-only) to train on all explicit ratings."
        ),
    )
    p_transfer.add_argument(
        "--rule-max-snipers",
        type=int,
        default=1,
        help="For rule-based coordinator: maximum number of snipers to assign per step when target rank is still > 5.",
    )
    p_transfer.add_argument("--target-device", default="cpu")
    p_transfer.add_argument("--target-lightgcn-layers", type=int, default=2)
    p_transfer.add_argument(
        "--use-segment-users-as-agents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use real segment user IDs (horror fans identified from clean data) as attack agents "
            "instead of creating fresh fake agent_N accounts. This fixes the cold-start isolation "
            "problem: segment users already have embeddings and graph edges in the target models, "
            "so adding item-1215 to their history directly promotes it within the real user graph."
        ),
    )
    p_transfer.add_argument(
        "--transfer-attack-roles",
        default="all",
        help=(
            "Comma-separated list of agent roles whose accepted interactions are injected into "
            "target model training. Default 'all' includes every accepted interaction. "
            "Use 'sniper' to inject only sniper-role interactions (target promotions only, "
            "no profiler/camouflaguer noise that creates accidental positives for other items)."
        ),
    )
    p_transfer.add_argument("--output", default="outputs/transfer_result.json")
    p_transfer.set_defaults(func=cmd_run_transfer)

    return parser


def main() -> int:
    """Entrypoint used by the console script in package metadata."""

    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
