"""Item-KNN recommender."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .base import BaseTargetRecommender, TargetModelConfig


@dataclass
class ItemKNNConfig(TargetModelConfig):
    """Item-KNN config placeholder."""


class ItemKNNRecommender(BaseTargetRecommender):
    """Item-item KNN based on cosine similarity of implicit interactions."""

    def __init__(self, config: Optional[ItemKNNConfig] = None):
        super().__init__(config=config or ItemKNNConfig())
        self.similarities: Optional[np.ndarray] = None
        self.topk: int = int(getattr(self.config, "knn_k", 50))

    def _fit_model(self, frame: pd.DataFrame) -> None:
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        mat = np.zeros((num_items, num_users), dtype=np.float32)
        for row in frame.itertuples(index=False):
            u = self.user_to_idx.get(str(row.user_id))
            i = self.item_to_idx.get(str(row.item_id))
            if u is None or i is None:
                continue
            mat[i, u] = 1.0

        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        mat_norm = mat / norms
        sims = mat_norm @ mat_norm.T
        np.fill_diagonal(sims, 0.0)

        if self.topk and self.topk < num_items:
            # keep only topK per item
            topk_idx = np.argpartition(-sims, self.topk, axis=1)[:, : self.topk]
            mask = np.zeros_like(sims, dtype=bool)
            rows = np.arange(num_items)[:, None]
            mask[rows, topk_idx] = True
            sims = sims * mask

        self.similarities = sims

    def _score_users_items(self, user_indices, item_indices):
        assert self.similarities is not None
        user_indices = user_indices.cpu().numpy()
        item_indices = item_indices.cpu().numpy()
        scores = []
        for u_idx in user_indices:
            user_id = self.idx_to_user[int(u_idx)]
            user_items = []
            if self.interactions is not None:
                user_items = self.interactions[self.interactions["user_id"] == user_id]["item_id"].astype(str).tolist()
            user_item_idx = [self.item_to_idx[i] for i in user_items if i in self.item_to_idx]
            if not user_item_idx:
                scores.append(np.zeros(len(item_indices), dtype=np.float32))
                continue
            sim_sum = self.similarities[item_indices][:, user_item_idx].sum(axis=1)
            scores.append(sim_sum)
        return torch.tensor(np.mean(np.stack(scores, axis=0), axis=0), device=self._device())
