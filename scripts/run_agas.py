#!/usr/bin/env python
"""Single entry point for the AGAS attack runner.

This script implements the canonical CLI described in the paper README:

    python scripts/run_agas.py \
        --dataset ml-100k \
        --victim lightgcn \
        --rounds 18 \
        --seed 42 \
        --n_workers 8 \
        --budget 0.01 \
        --out outputs/agas_ml100k_lightgcn_seed42.json

It is a thin wrapper around the underlying ``agas.cli`` CLI that builds a
Coordinator + worker pool, runs the AGAS round loop described in
``algorithms/agas_end_to_end.tex`` (with `--rounds` mapped to `T`), and writes
the full episode trace + summary metrics to ``--out``.

The script accepts the seven flags required by the paper protocol:

    --dataset    one of {ml-100k, ml-1m, genome2021, netflix, douban, amazon}
    --victim     one of the 11 victims listed in experiment.tex
    --rounds     number of AGAS rounds (default 18)
    --seed       random seed
    --n_workers  size of the fake-user pool
    --budget     per-user interaction budget L (paper: per-user budget)
    --out        output JSON path

Everything else uses paper defaults (OpenAI provider, gpt-5.1, ReAct workers,
ReFlexion-style Coordinator memory).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make sure the local src/ is importable when this file is run directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agas.cli import main as _agas_cli_main  # noqa: E402


# Dataset name → preprocessed directory name (matches data/loaders/ adapters).
_DATASET_TO_PROCESSED: dict[str, str] = {
    "ml-100k": "ml-100k",
    "ml-1m": "ml-1m",
    "ml-latest-small": "ml-latest-small",  # tiny ML-100K-style sample for smoke tests
    "genome2021": "genome2021",
    "netflix": "netflix",
    "douban": "douban",
    "amazon": "amazon",
}

# Victim name → recsys.targets backend (subset; matches experiment.tex).
_VICTIM_TO_BACKEND: dict[str, str] = {
    "mf": "mf",
    "bpr": "bpr",
    "neumf": "neumf",
    "gmf": "gmf",
    "ncf": "ncf",
    "ngcf": "ngcf",
    "lightgcn": "lightgcn",
    "simgcl": "simgcl",
    "xsimgcl": "xsimgcl",
    "egcf": "egcf",
    "lightccf": "lightccf",
    # The default surrogate is matched by the empty key — used when the user
    # only wants the lightweight surrogate.
    "surrogate": "surrogate",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one AGAS attack episode against a target victim recommender.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(_DATASET_TO_PROCESSED.keys()),
        help="Which preprocessed dataset to attack.",
    )
    parser.add_argument(
        "--victim",
        required=True,
        choices=sorted(_VICTIM_TO_BACKEND.keys()),
        help="Victim recommender family (see experiment.tex).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=18,
        help="Number of AGAS rounds T (paper default: 18).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Fake-user pool size |U_f|.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.01,
        help="Per-user interaction budget L as a fraction of users.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output JSON path for the episode trace + summary metrics.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed"),
        help="Root of preprocessed dataset folders (default: ./processed).",
    )
    parser.add_argument(
        "--target-item-id",
        type=str,
        default=None,
        help="Optional explicit target item ID; default = first cluster item.",
    )
    parser.add_argument(
        "--target-keyword",
        type=str,
        default="horror",
        help="Genre/category keyword used to define the target cluster.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=("openai", "ollama"),
        help="LLM provider for Coordinator and worker policies (default openai).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-5.1"),
        help="LLM model identifier (default: $OPENAI_MODEL or gpt-5.1).",
    )
    # ---- RQ4 ablation pass-throughs (forwarded verbatim to `agas.cli run-episode`) ----
    parser.add_argument(
        "--disable-roles",
        type=str,
        default=None,
        help="Ablate roles, e.g. 'sn' for w/o Sniper or 'pr' for w/o Profiler "
             "(comma-separated: pr,sn,ca,in).",
    )
    parser.add_argument(
        "--disable-strategies",
        type=str,
        default=None,
        help="Ablate strategies, e.g. 's5' for w/o Silent Slowdown "
             "(comma-separated: s1..s8).",
    )
    parser.add_argument(
        "--random-coordinator",
        action="store_true",
        default=False,
        help="w/o Coordinator: random role assignment instead of the LLM.",
    )
    parser.add_argument(
        "--disable-signals",
        action="store_true",
        default=False,
        help="w/o Signals: blank the engineered signals before the Coordinator.",
    )
    parser.add_argument(
        "--worker-llm-model",
        type=str,
        default=None,
        help="Backbone for the workers (size axis of the ablation heatmap). "
             "Defaults to --model.",
    )
    parser.add_argument(
        "--coordinator-agent-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Turn the coordinator cross-round memory on/off (w/o Memory row).",
    )
    parser.add_argument(
        "--profile-validator",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Turn the ProfileValidator guardrail on/off.",
    )
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args() if argv is None else _parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Map the paper flag set onto the legacy ``agas.cli run-episode`` command.
    cli_argv = [
        "run-episode",
        "--processed-root",
        str(args.processed_root),
        "--dataset",
        _DATASET_TO_PROCESSED[args.dataset],
        "--num-steps",
        str(args.rounds),
        "--num-agents",
        str(args.n_workers),
        "--seed",
        str(args.seed),
        "--target-keyword",
        args.target_keyword,
        "--coordinator-policy",
        ("openai" if args.provider == "openai" else "ollama"),
        "--worker-policy",
        ("openai" if args.provider == "openai" else "ollama"),
        "--llm-model",
        args.model,
        "--output",
        str(args.out),
        "--prompt-root",
        str(_REPO_ROOT / "prompts"),
    ]
    if args.target_item_id:
        cli_argv.extend(["--target-item-id", str(args.target_item_id)])

    # Forward the RQ4 ablation flags when the user set them.
    if args.disable_roles:
        cli_argv.extend(["--disable-roles", str(args.disable_roles)])
    if args.disable_strategies:
        cli_argv.extend(["--disable-strategies", str(args.disable_strategies)])
    if args.random_coordinator:
        cli_argv.append("--random-coordinator")
    if args.disable_signals:
        cli_argv.append("--disable-signals")
    if args.worker_llm_model:
        cli_argv.extend(["--worker-llm-model", str(args.worker_llm_model)])
    if args.coordinator_agent_memory is not None:
        cli_argv.append(
            "--coordinator-agent-memory" if args.coordinator_agent_memory
            else "--no-coordinator-agent-memory"
        )
    if args.profile_validator is not None:
        cli_argv.append(
            "--profile-validator" if args.profile_validator else "--no-profile-validator"
        )

    # Inject budget L as a CLI hint (legacy CLI uses --rule-max-snipers / etc.,
    # but we pass it through via env var so downstream tooling can pick it up).
    os.environ["AGAS_BUDGET"] = f"{args.budget:.4f}"

    # Override sys.argv so ``agas.cli.main`` parses our arguments.
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]] + cli_argv
        return int(_agas_cli_main() or 0)
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
