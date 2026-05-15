"""Recommender system utilities.

The torch-dependent classes from :mod:`agas.recsys.targets` are exposed
lazily through ``__getattr__`` so that ``import agas.recsys`` works without
the optional ``targets`` extra (install with ``pip install -e '.[targets]'``).
"""

from __future__ import annotations

from typing import Any

from agas.recsys.prepared_data import (
    filter_interactions_by_items,
    load_preprocessed_dataset,
    select_items_by_keyword,
)
from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig

__all__ = [
    "LightweightSurrogateRecommender",
    "SurrogateConfig",
    "TargetModelConfig",
    "NeuMFRecommender",
    "LightGCNRecommender",
    "load_preprocessed_dataset",
    "select_items_by_keyword",
    "filter_interactions_by_items",
]


def __getattr__(name: str) -> Any:
    """Forward lazy attribute access for torch-dependent recommenders."""

    if name in {
        "TargetModelConfig",
        "NeuMFRecommender",
        "LightGCNRecommender",
        "MFRecommender",
        "GMFRecommender",
        "BPRMFRecommender",
        "PMFRecommender",
        "WMFRecommender",
        "NMFRecommender",
        "NGCFRecommender",
        "GCMCRecommender",
        "ItemKNNRecommender",
        "SVDppRecommender",
        "IAutoRecRecommender",
        "UAutoRecRecommender",
        "CDAERecommender",
        "SequentialRecommender",
    }:
        from agas.recsys import targets

        return getattr(targets, name)
    raise AttributeError(f"module 'agas.recsys' has no attribute {name!r}")
