"""Minimal sequential recommender for transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .base import BaseTargetRecommender, TargetModelConfig


@dataclass
class SequentialConfig(TargetModelConfig):
    """Sequential-model config placeholder (inherits TargetModelConfig)."""


class SequentialRecommender(BaseTargetRecommender):
    """First-order Markov sequential recommender."""

    def __init__(self, config: Optional[SequentialConfig] = None):
        super().__init__(config=config or SequentialConfig())
        self._last_item_by_user: dict[int, int] = {}
        self._transition_probs: dict[int, dict[int, float]] = {}
        self._item_popularity: Optional[np.ndarray] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            self._last_item_by_user = {}
            self._transition_probs = {}
            self._item_popularity = np.zeros(len(self.item_to_idx), dtype=np.float32)
            return

        ordered = frame.copy()
        if "timestamp" in ordered.columns:
            ordered = ordered.sort_values(["user_id", "timestamp"], kind="mergesort")
        else:
            ordered = ordered.sort_values(["user_id"], kind="mergesort")

        user_ids = ordered["user_id"].astype(str).to_numpy()
        item_ids = ordered["item_id"].astype(str).to_numpy()

        # Popularity fallback
        pop = ordered.groupby("item_id").size()
        self._item_popularity = np.array([float(pop.get(i, 0.0)) for i in self.idx_to_item], dtype=np.float32)
        if self._item_popularity.sum() > 0:
            self._item_popularity /= self._item_popularity.sum()

        transitions: dict[int, dict[int, int]] = {}
        last_by_user: dict[int, int] = {}

        for u_str, i_str in zip(user_ids, item_ids):
            if u_str not in self.user_to_idx or i_str not in self.item_to_idx:
                continue
            u_idx = self.user_to_idx[u_str]
            i_idx = self.item_to_idx[i_str]
            prev = last_by_user.get(u_idx)
            if prev is not None:
                transitions.setdefault(prev, {}).setdefault(i_idx, 0)
                transitions[prev][i_idx] += 1
            last_by_user[u_idx] = i_idx

        self._last_item_by_user = last_by_user
        self._transition_probs = {}
        for prev, counts in transitions.items():
            total = float(sum(counts.values()))
            if total <= 0:
                continue
            self._transition_probs[prev] = {i: c / total for i, c in counts.items()}

    def _score_users_items(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        num_items = len(item_indices)
        if num_items == 0:
            return torch.zeros((len(user_indices), 0))

        item_idx_list = item_indices.cpu().numpy().tolist()
        scores = np.zeros((len(user_indices), num_items), dtype=np.float32)

        for row, u_idx in enumerate(user_indices.cpu().numpy().tolist()):
            last_item = self._last_item_by_user.get(int(u_idx))
            if last_item is not None and last_item in self._transition_probs:
                probs = self._transition_probs[last_item]
                for col, item_idx in enumerate(item_idx_list):
                    scores[row, col] = float(probs.get(int(item_idx), 0.0))
            elif self._item_popularity is not None:
                scores[row, :] = self._item_popularity[item_idx_list]

        return torch.tensor(scores, device=item_indices.device).mean(dim=0)
