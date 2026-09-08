"""Graph Convolutional Matrix Completion (GC-MC) recommender (simplified)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTargetRecommender, TargetModelConfig, _build_user_item_sets, _make_positive_pairs
from .mf import _sample_negatives


class _GCMC(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int, layers: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])

    def propagate(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        for layer in self.layers:
            all_emb = F.relu(layer(torch.sparse.mm(norm_adj, all_emb)))
        return all_emb[: self.user_emb.num_embeddings], all_emb[self.user_emb.num_embeddings :]


@dataclass
class GCMCConfig(TargetModelConfig):
    """GC-MC config placeholder."""


class GCMCRecommender(BaseTargetRecommender):
    """GC-MC target recommender wrapper."""

    def __init__(self, config: Optional[GCMCConfig] = None):
        super().__init__(config=config or GCMCConfig())
        self.model: Optional[_GCMC] = None
        self._norm_adj: Optional[torch.Tensor] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)

        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _GCMC(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim, 1).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)

        layers = int(getattr(self.config, "gcmc_layers", 1))
        model = _GCMC(num_users, num_items, self.config.embedding_dim, layers).to(device)
        self._norm_adj = self._build_norm_adj(frame, num_users, num_items, device)
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

                user_emb, item_emb = model.propagate(self._norm_adj)
                logits = (user_emb[users_t] * item_emb[items_t]).sum(dim=1)
                loss = loss_fn(logits, labels_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model = model

    def _build_norm_adj(self, frame: pd.DataFrame, num_users: int, num_items: int, device: torch.device) -> torch.Tensor:
        user_idx = frame["user_id"].map(self.user_to_idx).to_numpy()
        item_idx = frame["item_id"].map(self.item_to_idx).to_numpy()
        rows = np.concatenate([user_idx, item_idx + num_users])
        cols = np.concatenate([item_idx + num_users, user_idx])
        vals = np.ones_like(rows, dtype=np.float32)
        idx = torch.tensor([rows, cols], device=device)
        vals_t = torch.tensor(vals, device=device)
        size = num_users + num_items
        adj = torch.sparse_coo_tensor(idx, vals_t, (size, size))
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        d_mat = torch.diag(deg_inv_sqrt)
        norm_adj = torch.sparse.mm(torch.sparse.mm(adj, d_mat), d_mat)
        return norm_adj

    def _score_users_items(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        assert self.model is not None and self._norm_adj is not None
        user_emb, item_emb = self.model.propagate(self._norm_adj)
        users = user_emb[user_indices]
        items = item_emb[item_indices]
        scores = users @ items.T
        return scores.mean(dim=0)
