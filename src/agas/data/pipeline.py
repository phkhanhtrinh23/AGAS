"""End-to-end preprocessing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from agas.data.discovery import build_default_adapters, discover_available_adapters
from agas.data.loaders.base import DatasetAdapter
from agas.data.schema import DatasetStats


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""

    data_root: Path = Path("data")
    output_root: Path = Path("processed")
    chunk_size: int = 200_000
    max_rows_per_dataset: Optional[int] = 2_000_000
    overwrite: bool = True
    include_datasets: Optional[set[str]] = None


def _write_chunk(path: Path, chunk: pd.DataFrame, first: bool) -> None:
    """Append or create a CSV file while writing headers only once.

    Args:
        path: Output CSV path.
        chunk: DataFrame chunk to write.
        first: Whether this is the first write (controls overwrite/header).
    """

    chunk.to_csv(path, mode="w" if first else "a", index=False, header=first)


def preprocess_dataset(adapter: DatasetAdapter, cfg: PreprocessConfig) -> DatasetStats:
    """Preprocess one dataset adapter into canonical CSV files.

    Args:
        adapter: Dataset adapter that reads one raw dataset family.
        cfg: Pipeline configuration including roots and row limits.

    Returns:
        Summary statistics and output paths for the processed dataset.
    """

    dataset_dir = cfg.output_root / adapter.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = dataset_dir / "interactions.csv"
    items_path = dataset_dir / "items.csv"

    if cfg.overwrite:
        if interactions_path.exists():
            interactions_path.unlink()
        if items_path.exists():
            items_path.unlink()

    users_seen: set[str] = set()
    items_seen: set[str] = set()
    n_interactions = 0
    first = True

    for chunk in adapter.iter_interactions(
        data_root=cfg.data_root,
        max_rows=cfg.max_rows_per_dataset,
        chunk_size=cfg.chunk_size,
    ):
        _write_chunk(interactions_path, chunk, first=first)
        first = False
        n_interactions += len(chunk)
        users_seen.update(chunk["user_id"].dropna().astype(str).unique().tolist())
        items_seen.update(chunk["item_id"].dropna().astype(str).unique().tolist())

    items_df = adapter.load_items(cfg.data_root, max_rows=cfg.max_rows_per_dataset)
    items_df.to_csv(items_path, index=False)

    stats = DatasetStats(
        dataset=adapter.name,
        interactions=n_interactions,
        unique_users=len(users_seen),
        unique_items=len(items_seen) if items_seen else int(items_df["item_id"].nunique()),
        output_interactions=interactions_path,
        output_items=items_path,
        notes=(
            f"max_rows_per_dataset={cfg.max_rows_per_dataset}" if cfg.max_rows_per_dataset is not None else "full export"
        ),
    )

    with (dataset_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": stats.dataset,
                "interactions": stats.interactions,
                "unique_users": stats.unique_users,
                "unique_items": stats.unique_items,
                "interactions_path": str(stats.output_interactions),
                "items_path": str(stats.output_items),
                "notes": stats.notes,
            },
            f,
            indent=2,
        )

    return stats


def preprocess_all(
    cfg: PreprocessConfig,
    adapters: Optional[Iterable[DatasetAdapter]] = None,
) -> list[DatasetStats]:
    """Preprocess all available datasets from ``cfg.data_root``.

    Args:
        cfg: Pipeline configuration including roots and row limits.
        adapters: Optional adapter override; defaults to full registry.

    Returns:
        Per-dataset preprocessing statistics.
    """

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    registered = list(adapters) if adapters is not None else build_default_adapters()
    available = discover_available_adapters(cfg.data_root, registered)

    if cfg.include_datasets:
        include = {name.strip() for name in cfg.include_datasets}
        available = [adapter for adapter in available if adapter.name in include]

    stats = [preprocess_dataset(adapter, cfg) for adapter in available]

    summary = pd.DataFrame(
        [
            {
                "dataset": s.dataset,
                "interactions": s.interactions,
                "unique_users": s.unique_users,
                "unique_items": s.unique_items,
                "interactions_path": str(s.output_interactions),
                "items_path": str(s.output_items),
                "notes": s.notes,
            }
            for s in stats
        ]
    )
    summary.to_csv(cfg.output_root / "summary.csv", index=False)
    return stats


def preprocess_all_datasets(
    data_root: Path,
    output_root: Path,
    include: Optional[Iterable[str]] = None,
    max_rows_per_dataset: Optional[int] = 2_000_000,
    chunk_size: int = 200_000,
) -> list[DatasetStats]:
    """Compatibility wrapper used by CLI/tests.

    Args:
        data_root: Raw dataset root directory.
        output_root: Processed dataset output directory.
        include: Optional iterable of dataset names to include.
        max_rows_per_dataset: Optional per-dataset row cap.
        chunk_size: CSV/record chunk size used during preprocessing.

    Returns:
        Per-dataset preprocessing statistics.
    """

    cfg = PreprocessConfig(
        data_root=data_root,
        output_root=output_root,
        chunk_size=chunk_size,
        max_rows_per_dataset=max_rows_per_dataset,
        include_datasets=set(include) if include else None,
    )
    return preprocess_all(cfg)


def load_interactions(output_root: Path, dataset: str) -> pd.DataFrame:
    """Load canonical interactions for one preprocessed dataset.

    Args:
        output_root: Root directory containing processed datasets.
        dataset: Dataset subdirectory name.

    Returns:
        Canonical interactions DataFrame.
    """

    path = output_root / dataset / "interactions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed interactions: {path}")
    return pd.read_csv(path)


def load_items(output_root: Path, dataset: str) -> pd.DataFrame:
    """Load canonical items for one preprocessed dataset.

    Args:
        output_root: Root directory containing processed datasets.
        dataset: Dataset subdirectory name.

    Returns:
        Canonical items DataFrame.
    """

    path = output_root / dataset / "items.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed items: {path}")
    return pd.read_csv(path)
