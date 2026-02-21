"""Helpers for reading canonical preprocessed data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_preprocessed_dataset(processed_root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load interactions/items CSV files for one canonical dataset."""

    base = processed_root / dataset
    interactions = pd.read_csv(base / "interactions.csv")
    items = pd.read_csv(base / "items.csv")

    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    items["item_id"] = items["item_id"].astype(str)
    return interactions, items


def select_items_by_keyword(items: pd.DataFrame, keyword: str, column: str = "genres") -> list[str]:
    """Return item IDs where ``column`` contains ``keyword`` (case-insensitive)."""

    key = keyword.lower().strip()
    mask = items[column].fillna("").astype(str).str.lower().str.contains(key)
    return items.loc[mask, "item_id"].astype(str).tolist()


def filter_interactions_by_items(interactions: pd.DataFrame, item_ids: Iterable[str]) -> pd.DataFrame:
    """Return interactions that involve the given item IDs."""

    item_set = set(map(str, item_ids))
    return interactions.loc[interactions["item_id"].astype(str).isin(item_set)].copy()
