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
from agas.agents.worker import build_worker_pool
from agas.data.pipeline import PreprocessConfig, preprocess_all
from agas.llm.providers import build_llm_client
from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig
from agas.simulation.environment import AGASEnvironment, DefenseConfig
from agas.simulation.episode import AGASEpisodeRunner, EpisodeConfig, default_agent_ids


def _parse_optional_int(raw: str | None) -> Optional[int]:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"none", "null", "all", "full"}:
        return None
    return int(value)


def _read_limited_csv(path: Path, max_rows: Optional[int] = None, chunksize: int = 200_000) -> pd.DataFrame:
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


def cmd_preprocess(args: argparse.Namespace) -> int:
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
    model, interactions, items = _fit_surrogate_from_processed(
        processed_root=Path(args.processed_root),
        dataset=args.dataset,
        max_interactions=_parse_optional_int(args.max_interactions),
        n_factors=args.n_factors,
    )

    agent_ids = default_agent_ids(args.num_agents)
    workers = build_worker_pool(agent_ids)

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
        policy = LLMCoordinatorPolicy(client=client, agent_order=agent_ids)

    coordinator = Coordinator(policy=policy)

    env = AGASEnvironment(
        recommender=model,
        base_interactions=interactions,
        items=items,
        target_item_id=str(args.target_item_id),
        target_keyword=args.target_keyword,
        defense_config=DefenseConfig(
            spike_threshold=args.spike_threshold,
            lockdown_drop_prob=args.lockdown_drop_prob,
        ),
        seed=args.seed,
    )

    runner = AGASEpisodeRunner(
        coordinator=coordinator,
        environment=env,
        workers=workers,
        config=EpisodeConfig(num_steps=args.num_steps, num_workers=args.num_agents),
    )
    result = runner.run()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "target_item_id": str(args.target_item_id),
                "target_keyword": args.target_keyword,
                "num_steps": args.num_steps,
                "num_agents": args.num_agents,
                "coordinator_policy": args.coordinator_policy,
                "final_rank": result.final_rank,
                "final_total_candidates": result.final_total_candidates,
                "final_worker_states": result.final_worker_states,
                "history": result.history,
            },
            f,
            indent=2,
        )

    print(f"Episode finished. Target rank: {result.final_rank}/{result.final_total_candidates}")
    print(f"Saved detailed timeline to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
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
    p_run.add_argument("--seed", type=int, default=42)

    p_run.add_argument("--coordinator-policy", choices=["rule", "openai", "ollama"], default="rule")
    p_run.add_argument("--llm-model", default="gpt-4o-mini")
    p_run.add_argument("--openai-api-key", default=None)
    p_run.add_argument("--ollama-host", default="http://localhost:11434")

    p_run.add_argument("--spike-threshold", type=int, default=2)
    p_run.add_argument("--lockdown-drop-prob", type=float, default=0.55)

    p_run.add_argument("--output", default="outputs/episode_result.json")
    p_run.set_defaults(func=cmd_run_episode)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
