"""Simplified SVD++ recommender."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from .base import BaseTargetRecommender, TargetModelConfig, _build_user_item_sets, _make_positive_pairs
from .mf import _sample_negatives


class _SVDpp(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.item_implicit = nn.Embedding(num_items, embed_dim)

    def forward(self, users: torch.Tensor, items: torch.Tensor, implicit_sum: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(users) + implicit_sum
        i = self.item_emb(items)
        return (u * i).sum(dim=1)


@dataclass
class SVDppConfig(TargetModelConfig):
    """SVD++ config placeholder."""


class SVDppRecommender(BaseTargetRecommender):
    """SVD++ with implicit feedback aggregation and BCE loss."""

    def __init__(self, config: Optional[SVDppConfig] = None):
        super().__init__(config=config or SVDppConfig())
        self.model: Optional[_SVDpp] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)
        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _SVDpp(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        model = _SVDpp(num_users, num_items, self.config.embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        user_hist_indices: list[list[int]] = [[] for _ in range(num_users)]
        for user_id, items in user_items.items():
            if user_id not in self.user_to_idx:
                continue
            u_idx = self.user_to_idx[user_id]
            for item_id in items:
                if item_id in self.item_to_idx:
                    user_hist_indices[u_idx].append(self.item_to_idx[item_id])

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
                for u_idx, _ in batch:
                    neg_items.extend(
                        _sample_negatives(self._rng, user_items, self.idx_to_item, self.idx_to_user, num_items, int(u_idx), self.config.num_negatives)
                    )
                users_rep = np.repeat(users, self.config.num_negatives)
                all_users = np.concatenate([users, users_rep])
                all_items = np.concatenate([pos_items, np.array(neg_items, dtype=np.int64)])
                labels = np.concatenate([
                    np.ones(len(users), dtype=np.float32),
                    np.zeros(len(neg_items), dtype=np.float32),
                ])

                users_t = torch.tensor(all_users, device=device)
                items_t = torch.tensor(all_items, device=device)
                labels_t = torch.tensor(labels, device=device)

                implicit_sum = []
                for u_idx in users_t.tolist():
                    hist = user_hist_indices[int(u_idx)]
                    if not hist:
                        implicit_sum.append(torch.zeros(self.config.embedding_dim, device=device))
                        continue
                    hist_t = torch.tensor(hist, device=device)
                    emb = model.item_implicit(hist_t).sum(dim=0)
                    emb = emb / np.sqrt(len(hist))
                    implicit_sum.append(emb)
                implicit_sum = torch.stack(implicit_sum, dim=0)

                optimizer.zero_grad()
                logits = model(users_t, items_t, implicit_sum)
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
            u_idx = int(u)
            hist = []
            if u_idx < len(self.idx_to_user):
                user_id = self.idx_to_user[u_idx]
                if self.interactions is not None:
                    hist = self.interactions[self.interactions["user_id"] == user_id]["item_id"].astype(str).tolist()
            hist_idx = [self.item_to_idx[i] for i in hist if i in self.item_to_idx]
            if hist_idx:
                hist_t = torch.tensor(hist_idx, device=device)
                implicit_sum = self.model.item_implicit(hist_t).sum(dim=0) / np.sqrt(len(hist_idx))
            else:
                implicit_sum = torch.zeros(self.config.embedding_dim, device=device)
            u_expand = u.repeat(len(items))
            implicit_expand = implicit_sum.unsqueeze(0).repeat(len(items), 1)
            logits = self.model(u_expand, items, implicit_expand)
            scores.append(logits)
        return torch.stack(scores, dim=0).mean(dim=0)
