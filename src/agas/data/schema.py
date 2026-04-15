"""Canonical dataset schemas for preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

INTERACTION_COLUMNS = [
    "dataset",
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "split",
    "context_json",
    "source",
]

ITEM_COLUMNS = [
    "dataset",
    "item_id",
    "title",
    "genres",
    "category",
    "metadata_json",
    "source",
]


@dataclass
class DatasetStats:
    """Summary metrics and output paths produced by preprocessing one dataset."""

    dataset: str
    interactions: int
    unique_users: int
    unique_items: int
    output_interactions: Path
    output_items: Path
    notes: Optional[str] = None


def finalize_interactions(df: pd.DataFrame, dataset: str, source: str) -> pd.DataFrame:
    """Normalize arbitrary interaction columns to canonical AGAS interaction schema.

    Args:
        df: Input DataFrame containing interaction-like columns.
        dataset: Canonical dataset name to stamp into output rows.
        source: Source file identifier for traceability.

    Returns:
        Canonical interactions DataFrame with required columns and dtypes.
    """

    frame = df.copy()
    frame["dataset"] = dataset
    frame["source"] = source

    for col in INTERACTION_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA

    frame = frame[INTERACTION_COLUMNS]
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").astype("Int64")
    frame["split"] = frame["split"].astype("string")
    frame["context_json"] = frame["context_json"].astype("string")
    return frame.dropna(subset=["user_id", "item_id", "rating"])


def finalize_items(df: pd.DataFrame, dataset: str, source: str) -> pd.DataFrame:
    """Normalize arbitrary item metadata to canonical AGAS item schema.

    Args:
        df: Input DataFrame containing item metadata columns.
        dataset: Canonical dataset name to stamp into output rows.
        source: Source file identifier for traceability.

    Returns:
        Canonical items DataFrame with required columns and deduped item IDs.
    """

    frame = df.copy()
    frame["dataset"] = dataset
    frame["source"] = source

    for col in ITEM_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA

    frame = frame[ITEM_COLUMNS]
    frame["item_id"] = frame["item_id"].astype(str)
    frame["title"] = frame["title"].astype("string")
    frame["genres"] = frame["genres"].astype("string")
    frame["category"] = frame["category"].astype("string")
    frame["metadata_json"] = frame["metadata_json"].astype("string")
    return frame.drop_duplicates(subset=["dataset", "item_id"])
