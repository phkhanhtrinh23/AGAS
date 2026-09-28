#!/usr/bin/env python
"""Aggregate AGAS run outputs into the per-(dataset, victim) numbers used by the
paper's tables and figures.

The AGAS engine writes ONE JSON per run (per dataset / victim / seed / config).
The paper's tables and figures report *aggregates* of those runs:
``mean +/- 95% CI over 5 seeds``, and the promotion metrics are scaled by
``10^3`` (a cell ``40.0 +- 0.2`` means HR@10 = 0.0400 +- 0.0002).

This script is the bridge "runs -> numbers". It:

1. walks an output directory (recursively) and loads every ``*.json``;
2. understands both output schemas produced by the CLI
   (``run-episode`` and ``run-transfer``, see ``src/agas/cli.py``);
3. derives HR@K / NDCG@K from the reported target-item rank using the exact
   same closed form the CLI uses (``_rank_to_hr_ndcg``);
4. groups runs by ``(dataset, victim)`` and reports mean +/- 95% CI over seeds,
   in the ``x 10^3`` convention of the paper;
5. optionally writes a tidy CSV that a plotting script can read.

It changes nothing in the ``outputs/`` tree — it only reads.

Usage
-----
    python scripts/aggregate_results.py outputs/multiseed_welltrained
    python scripts/aggregate_results.py outputs/rq1_performance --k 10 --csv summary.csv
    python scripts/aggregate_results.py outputs/rq5_efficiency --tokens
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# rank -> HR@K / NDCG@K  (identical to agas.cli._rank_to_hr_ndcg)
# ---------------------------------------------------------------------------
def rank_to_hr_ndcg(rank: int, k: int) -> tuple[float, float]:
    """Return (HR@K, NDCG@K) for a single relevant item at 1-based ``rank``."""
    if rank <= 0 or k <= 0 or rank > k:
        return 0.0, 0.0
    return 1.0, float(1.0 / math.log2(rank + 1.0))


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Return (mean, half-width of the 95% CI) for a small sample.

    Uses the normal approximation ``1.96 * s / sqrt(n)``.  With n=1 the CI is 0.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(max(0.0, var))
    return (mean, 1.96 * std / math.sqrt(n))


# ---------------------------------------------------------------------------
# One (dataset, victim, seed) record, whatever the source schema.
# ---------------------------------------------------------------------------
class Record:
    __slots__ = (
        "dataset",
        "victim",
        "seed",
        "config",
        "final_rank",
        "initial_rank",
        "best_rank",
        "total",
        "hr",
        "ndcg",
        "tokens",
        "runtime",
    )

    def __init__(self, **kw: Any) -> None:
        for s in self.__slots__:
            setattr(self, s, kw.get(s))


def _seed_from_name(path: Path, fallback: Optional[int]) -> Optional[int]:
    """Best-effort seed extraction from a filename like ``..._seed42.json``."""
    stem = path.stem.lower()
    if "seed" in stem:
        tail = stem.split("seed")[-1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return int(digits)
    return fallback


def _iter_transfer_records(doc: dict, path: Path, k: int) -> Iterable[Record]:
    """Yield one Record per target victim for a ``run-transfer`` document."""
    dataset = str(doc.get("dataset", "?"))
    seed = doc.get("seed")
    config = path.parent.name

    for block_name in ("option_b", "option_a"):
        block = doc.get(block_name) or {}
        for victim, stats in block.items():
            if not isinstance(stats, dict):
                continue
            final_rank = stats.get("final_rank")
            initial_rank = stats.get("initial_rank")
            best_rank = stats.get("best_rank", final_rank)
            total = stats.get("final_total_candidates") or stats.get("initial_total_candidates")
            # Prefer the closed-form HR/NDCG from the reported rank so the number
            # is well defined even when the JSON pre-dates hr_at_k.
            use_rank = best_rank if best_rank is not None else final_rank
            hr = ndcg = None
            if use_rank is not None:
                hr, ndcg = rank_to_hr_ndcg(int(use_rank), k)
            yield Record(
                dataset=dataset,
                victim=str(victim),
                seed=_seed_from_name(path, seed),
                config=config,
                final_rank=final_rank,
                initial_rank=initial_rank,
                best_rank=best_rank,
                total=total,
                hr=hr,
                ndcg=ndcg,
                tokens=None,
            )
        if block:  # option_b wins if present; do not double count
            return


def _iter_episode_records(doc: dict, path: Path, k: int) -> Iterable[Record]:
    """Yield one Record for a ``run-episode`` document."""
    dataset = str(doc.get("dataset", "?"))
    final_rank = doc.get("final_rank")
    initial_rank = doc.get("initial_rank")
    best_rank = doc.get("best_rank", final_rank)
    total = doc.get("final_total_candidates")
    use_rank = best_rank if best_rank is not None else final_rank
    hr = ndcg = None
    if use_rank is not None:
        hr, ndcg = rank_to_hr_ndcg(int(use_rank), k)
    tok = ((doc.get("token_usage") or {}).get("aggregate") or {}).get("total_tokens")
    runtime = doc.get("runtime_sec")
    yield Record(
        dataset=dataset,
        victim="episode",
        seed=_seed_from_name(path, None),
        config=path.parent.name,
        final_rank=final_rank,
        initial_rank=initial_rank,
        best_rank=best_rank,
        total=total,
        hr=hr,
        ndcg=ndcg,
        tokens=tok,
        runtime=runtime,
    )


def load_records(root: Path, k: int) -> list[Record]:
    records: list[Record] = []
    for path in sorted(root.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(doc, dict):
            continue
        if "option_a" in doc or "option_b" in doc or "transfer_mode" in doc:
            records.extend(_iter_transfer_records(doc, path, k))
        elif "final_rank" in doc:
            records.extend(_iter_episode_records(doc, path, k))
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Directory of run JSONs (searched recursively).")
    ap.add_argument("--k", type=int, default=10, help="K for HR@K / NDCG@K (default 10).")
    ap.add_argument("--scale", type=float, default=1000.0, help="Multiply HR/NDCG by this (paper uses 1e3).")
    ap.add_argument("--tokens", action="store_true", help="Also report mean total tokens per run.")
    ap.add_argument("--timing", action="store_true", help="Also report mean wall-clock seconds per run.")
    ap.add_argument("--csv", type=Path, default=None, help="Optional tidy CSV output path.")
    args = ap.parse_args()

    records = load_records(args.root, args.k)
    if not records:
        print(f"No usable run JSONs found under {args.root}")
        return 1

    groups: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.config, r.dataset, r.victim)].append(r)

    header = (
        f"{'config':<22}{'dataset':<16}{'victim':<12}{'n':>3}  "
        f"{'HR@%d(x%g)' % (args.k, args.scale):>16}{'NDCG@%d(x%g)' % (args.k, args.scale):>18}"
        f"{'rank_delta':>14}"
    )
    if args.tokens:
        header += f"{'tokens':>14}"
    if args.timing:
        header += f"{'sec/run':>12}"
    print(header)
    print("-" * len(header))

    rows_for_csv: list[dict] = []
    for key in sorted(groups):
        config, dataset, victim = key
        rs = groups[key]
        hr_vals = [r.hr * args.scale for r in rs if r.hr is not None]
        ndcg_vals = [r.ndcg * args.scale for r in rs if r.ndcg is not None]
        delta_vals = [
            float(r.initial_rank - r.final_rank)
            for r in rs
            if r.initial_rank is not None and r.final_rank is not None
        ]
        hr_m, hr_ci = mean_ci(hr_vals)
        nd_m, nd_ci = mean_ci(ndcg_vals)
        dl_m, dl_ci = mean_ci(delta_vals)
        line = (
            f"{config:<22}{dataset:<16}{victim:<12}{len(rs):>3}  "
            f"{hr_m:>7.2f}+-{hr_ci:<6.2f}{nd_m:>9.2f}+-{nd_ci:<6.2f}"
            f"{dl_m:>8.1f}+-{dl_ci:<4.1f}"
        )
        row = {
            "config": config,
            "dataset": dataset,
            "victim": victim,
            "n": len(rs),
            f"hr@{args.k}_x{int(args.scale)}_mean": round(hr_m, 4),
            f"hr@{args.k}_x{int(args.scale)}_ci": round(hr_ci, 4),
            f"ndcg@{args.k}_x{int(args.scale)}_mean": round(nd_m, 4),
            f"ndcg@{args.k}_x{int(args.scale)}_ci": round(nd_ci, 4),
            "rank_delta_mean": round(dl_m, 3),
            "rank_delta_ci": round(dl_ci, 3),
        }
        if args.tokens:
            tok_vals = [float(r.tokens) for r in rs if r.tokens is not None]
            tok_m, tok_ci = mean_ci(tok_vals)
            line += f"{tok_m:>10.0f}+-{tok_ci:<3.0f}" if tok_vals else f"{'--':>14}"
            row["tokens_mean"] = round(tok_m, 1) if tok_vals else None
        if args.timing:
            rt_vals = [float(r.runtime) for r in rs if r.runtime is not None]
            rt_m, rt_ci = mean_ci(rt_vals)
            line += f"{rt_m:>9.1f}+-{rt_ci:<2.0f}" if rt_vals else f"{'--':>12}"
            row["runtime_sec_mean"] = round(rt_m, 2) if rt_vals else None
        print(line)
        rows_for_csv.append(row)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"\nWrote tidy summary to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
