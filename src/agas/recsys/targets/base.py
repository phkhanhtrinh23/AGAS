"""Shared utilities for target recommender backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass
class TargetModelConfig:
    """Hyperparameters for target recommender models."""

    embedding_dim: int = 32
    epochs: int = 3
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_negatives: int = 4
    positive_threshold: float = 4.0
    seed: int = 42
    device: str = "cpu"
    lightgcn_layers: int = 2


class BaseTargetRecommender:
    """Base class for target recommenders used for transfer evaluation."""

    def __init__(self, config: TargetModelConfig | None = None):
        self.config = config or TargetModelConfig()
        self.interactions: Optional[pd.DataFrame] = None
        self.items: Optional[pd.DataFrame] = None
        self.user_to_idx: dict[str, int] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_user: list[str] = []
        self.idx_to_item: list[str] = []
        self.global_mean: float = 0.0
        self.item_bias: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(self.config.seed)

    def fit(self, interactions: pd.DataFrame, items: Optional[pd.DataFrame] = None) -> "BaseTargetRecommender":
        """Fit the target model on explicit interactions.

        Args:
            interactions: Canonical interactions with columns ``user_id``, ``item_id``, ``rating``.
            items: Optional item metadata table.

        Returns:
            Self for chaining.
        """

        frame = interactions[["user_id", "item_id", "rating"]].copy()
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)
        self.interactions = frame
        self.items = items.copy() if items is not None else None

        users = frame["user_id"].drop_duplicates().tolist()
        items_list = frame["item_id"].drop_duplicates().tolist()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: j for j, i in enumerate(items_list)}
        self.idx_to_user = list(users)
        self.idx_to_item = list(items_list)

        self.global_mean = float(frame["rating"].mean()) if len(frame) else 0.0
        bias = frame.groupby("item_id")["rating"].mean()
        self.item_bias = np.array([float(bias.get(i, self.global_mean)) for i in self.idx_to_item])

        self._fit_model(frame)
        return self

    def append_interactions(self, new_interactions: pd.DataFrame, refit: bool = True) -> None:
        """Append interactions and optionally refit the model.

        Args:
            new_interactions: New interaction rows.
            refit: Whether to refit immediately.
        """

        if self.interactions is None:
            self.fit(new_interactions, self.items)
            return

        appended = pd.concat(
            [self.interactions, new_interactions[["user_id", "item_id", "rating"]]], ignore_index=True
        )
        if refit:
            self.fit(appended, self.items)
        else:
            self.interactions = appended

    def rank_item(
        self,
        item_id: str,
        segment_user_ids: Optional[Iterable[str]] = None,
        candidate_items: Optional[Sequence[str]] = None,
    ) -> tuple[int, int]:
        """Return 1-based rank of ``item_id`` among candidates.

        Args:
            item_id: Target item ID to rank.
            segment_user_ids: Optional user subset used for segment averaging.
            candidate_items: Optional candidate item subset for ranking.

        Returns:
            Tuple ``(rank, total_candidates)`` with 1-based rank.
        """

        if self.interactions is None:
            raise RuntimeError("Model is not fit")

        item_id = str(item_id)
        if candidate_items is None:
            candidate_items = self.idx_to_item
        else:
            candidate_items = [str(i) for i in candidate_items if str(i) in self.item_to_idx]

        if not candidate_items:
            return (0, 0)

        user_indices = self._segment_user_indices(segment_user_ids)
        item_indices = torch.tensor([self.item_to_idx[i] for i in candidate_items], device=self._device())

        with torch.no_grad():
            scores = self._score_users_items(user_indices, item_indices)
        order = torch.argsort(scores, descending=True).cpu().numpy().tolist()
        ordered_items = [candidate_items[i] for i in order]
        for pos, cur in enumerate(ordered_items, start=1):
            if cur == item_id:
                return (pos, len(ordered_items))
        return (len(ordered_items) + 1, len(ordered_items))

    def mean_scores_for_segment(
        self,
        segment_user_ids: Optional[Iterable[str]] = None,
        candidate_items: Optional[Sequence[str]] = None,
    ) -> dict[str, float]:
        """Compute mean scores for a user segment over candidate items.

        Args:
            segment_user_ids: Optional user subset used for segment averaging.
            candidate_items: Optional candidate item subset to score.

        Returns:
            Mapping ``item_id -> mean score`` aligned with target model outputs.
        """

        if self.interactions is None:
            raise RuntimeError("Model is not fit")

        if candidate_items is None:
            candidate_items = self.idx_to_item
        else:
            candidate_items = [str(i) for i in candidate_items if str(i) in self.item_to_idx]

        if not candidate_items:
            return {}

        user_indices = self._segment_user_indices(segment_user_ids)
        item_indices = torch.tensor([self.item_to_idx[i] for i in candidate_items], device=self._device())
        with torch.no_grad():
            scores = self._score_users_items(user_indices, item_indices).cpu().numpy()

        return {item_id: float(score) for item_id, score in zip(candidate_items, scores)}

    def _segment_user_indices(self, segment_user_ids: Optional[Iterable[str]]) -> torch.Tensor:
        if segment_user_ids:
            indices = [self.user_to_idx[u] for u in map(str, segment_user_ids) if u in self.user_to_idx]
            if not indices:
                indices = list(range(len(self.idx_to_user)))
        else:
            indices = list(range(len(self.idx_to_user)))
        return torch.tensor(indices, device=self._device())

    def _device(self) -> torch.device:
        return torch.device(self.config.device)

    def _fit_model(self, frame: pd.DataFrame) -> None:
        raise NotImplementedError

    def _score_users_items(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _build_user_item_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    user_items: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        user_items.setdefault(str(row.user_id), set()).add(str(row.item_id))
    return user_items


def _make_positive_pairs(frame: pd.DataFrame, threshold: float) -> list[tuple[str, str]]:
    positives = frame[frame["rating"] >= threshold]
    return [(str(r.user_id), str(r.item_id)) for r in positives.itertuples(index=False)]
