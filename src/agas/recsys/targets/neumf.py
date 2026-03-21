"""Minimal NeuMF implementation for transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from .base import BaseTargetRecommender, TargetModelConfig, _build_user_item_sets, _make_positive_pairs


class _NeuMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int):
        super().__init__()
        self.user_gmf = nn.Embedding(num_users, embed_dim)
        self.item_gmf = nn.Embedding(num_items, embed_dim)
        self.user_mlp = nn.Embedding(num_users, embed_dim)
        self.item_mlp = nn.Embedding(num_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(embed_dim * 2, 1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.user_gmf(users) * self.item_gmf(items)
        mlp_in = torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1)
        mlp = self.mlp(mlp_in)
        concat = torch.cat([gmf, mlp], dim=-1)
        return self.out(concat).squeeze(-1)


@dataclass
class NeuMFConfig(TargetModelConfig):
    """NeuMF-specific config placeholder (inherits TargetModelConfig)."""


class NeuMFRecommender(BaseTargetRecommender):
    """NeuMF target recommender wrapper."""

    def __init__(self, config: Optional[NeuMFConfig] = None):
        super().__init__(config=config or NeuMFConfig())
        self.model: Optional[_NeuMF] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)

        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _NeuMF(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)

        model = _NeuMF(num_users, num_items, self.config.embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        pos_pairs = [(self.user_to_idx[u], self.item_to_idx[i]) for u, i in positives if u in self.user_to_idx and i in self.item_to_idx]
        pos_pairs = np.array(pos_pairs, dtype=np.int64)
        if len(pos_pairs) == 0:
            self.model = model
            return

        batch_size = max(1, int(self.config.batch_size))
        for _ in range(int(self.config.epochs)):
            self._rng.shuffle(pos_pairs)
            for start in range(0, len(pos_pairs), batch_size):
                batch = pos_pairs[start : start + batch_size]
                users = batch[:, 0]
                pos_items = batch[:, 1]

                neg_items = []
                for u_idx, pos_i in batch:
                    user_id = self.idx_to_user[int(u_idx)]
                    seen = user_items.get(user_id, set())
                    for _ in range(self.config.num_negatives):
                        j = self._rng.integers(0, num_items)
                        while self.idx_to_item[int(j)] in seen:
                            j = self._rng.integers(0, num_items)
                        neg_items.append(j)

                users_rep = np.repeat(users, self.config.num_negatives)
                pos_users = np.concatenate([users, users_rep])
                pos_items_full = np.concatenate([pos_items, np.array(neg_items, dtype=np.int64)])
                labels = np.concatenate([
                    np.ones(len(users), dtype=np.float32),
                    np.zeros(len(neg_items), dtype=np.float32),
                ])

                users_t = torch.tensor(pos_users, device=device)
                items_t = torch.tensor(pos_items_full, device=device)
                labels_t = torch.tensor(labels, device=device)

                optimizer.zero_grad()
                logits = model(users_t, items_t)
                loss = loss_fn(logits, labels_t)
                loss.backward()
                optimizer.step()

        self.model = model

    def _score_users_items(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        device = self._device()
        users = user_indices.to(device)
        items = item_indices.to(device)
        scores = []
        for u in users:
            u_expand = u.repeat(len(items))
            logits = self.model(u_expand, items)
            scores.append(logits)
        stacked = torch.stack(scores, dim=0)
        return stacked.mean(dim=0)
