"""Command line interface for AGAS project."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from agas.agents.coordinator import Coordinator, LLMCoordinatorPolicy, RuleBasedCoordinatorPolicy
from agas.llm.prompt_store import PromptStore
from agas.agents.worker import build_worker_pool
from agas.data.pipeline import PreprocessConfig, preprocess_all
from agas.llm.providers import build_llm_client
from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig
from agas.simulation.environment import AGASEnvironment, DefenseConfig
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

    interactions_path = processed_root / dataset / "interactions.csv"
    items_path = processed_root / dataset / "items.csv"
    if not interactions_path.exists() or not items_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessed files for dataset '{dataset}'. Run preprocess first."
        )

    interactions = _read_limited_csv(interactions_path, max_rows=max_interactions)
    items = pd.read_csv(items_path)

    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions = interactions.dropna(subset=["user_id", "item_id", "rating"])

    model = LightweightSurrogateRecommender(config=SurrogateConfig(n_factors=n_factors))
    model.set_items(items)
    model.fit(interactions)
    return model, interactions, items


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

    if args.coordinator_policy == "rule":
        policy = RuleBasedCoordinatorPolicy(agent_order=agent_ids)
    else:
        if args.coordinator_policy == "openai":
            api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI coordinator policy")
            client = build_llm_client(provider="openai", model=args.llm_model, api_key=api_key)
        else:
            client = build_llm_client(provider="ollama", model=args.llm_model, host=args.ollama_host)
        policy = LLMCoordinatorPolicy(client=client, agent_order=agent_ids, prompt_store=prompt_store)

    coordinator = Coordinator(policy=policy)
    worker_policy_name = args.worker_policy
    worker_llm_client = None
    if args.worker_policy != "rule":
        worker_model = args.worker_llm_model or args.llm_model
        if args.worker_policy == "openai":
            api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI worker policy")
            worker_llm_client = build_llm_client(provider="openai", model=worker_model, api_key=api_key)
        else:
            worker_llm_client = build_llm_client(provider="ollama", model=worker_model, host=args.ollama_host)
    workers = build_worker_pool(
        agent_ids,
        llm_client=worker_llm_client,
        prompt_store=prompt_store,
        policy_name=worker_policy_name,
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

    p_run.add_argument("--coordinator-policy", choices=["rule", "openai", "ollama"], default="openai")
    p_run.add_argument("--llm-model", default="gpt-5-mini")
    p_run.add_argument("--openai-api-key", default=None)
    p_run.add_argument("--ollama-host", default="http://localhost:11434")
    p_run.add_argument("--worker-policy", choices=["rule", "openai", "ollama"], default="rule")
    p_run.add_argument("--worker-llm-model", default=None)
    p_run.add_argument("--prompt-root", default="prompts")
    p_run.add_argument("--expose-defense-state", action="store_true")

    p_run.add_argument("--spike-threshold", type=int, default=2)
    p_run.add_argument("--lockdown-drop-prob", type=float, default=0.55)

    p_run.add_argument("--output", default="outputs/episode_result.json")
    p_run.set_defaults(func=cmd_run_episode)

    return parser


def main() -> int:
    """Entrypoint used by the console script in package metadata."""

    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
