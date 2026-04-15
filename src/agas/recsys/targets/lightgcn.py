"""Minimal LightGCN implementation for transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from .base import BaseTargetRecommender, TargetModelConfig, _build_user_item_sets, _make_positive_pairs


class _LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int, layers: int):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.layers = layers
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

    def propagate(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[: self.num_users], out[self.num_users :]


@dataclass
class LightGCNConfig(TargetModelConfig):
    """LightGCN-specific config placeholder (inherits TargetModelConfig)."""


class LightGCNRecommender(BaseTargetRecommender):
    """LightGCN target recommender wrapper."""

    def __init__(self, config: Optional[LightGCNConfig] = None):
        super().__init__(config=config or LightGCNConfig())
        self.model: Optional[_LightGCN] = None
        self._norm_adj: Optional[torch.Tensor] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)

        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _LightGCN(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim, self.config.lightgcn_layers).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)

        model = _LightGCN(num_users, num_items, self.config.embedding_dim, self.config.lightgcn_layers).to(device)
        self._norm_adj = self._build_norm_adj(frame, num_users, num_items, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

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
                    explicit_negs = self._explicit_neg_pairs.get(int(u_idx), [])
                    if explicit_negs:
                        neg_items.append(int(explicit_negs[self._rng.integers(0, len(explicit_negs))]))
                    else:
                        j = self._rng.integers(0, num_items)
                        while self.idx_to_item[int(j)] in seen:
                            j = self._rng.integers(0, num_items)
                        neg_items.append(j)

                users_t = torch.tensor(users, device=device)
                pos_t = torch.tensor(pos_items, device=device)
                neg_t = torch.tensor(neg_items, device=device)

                user_emb, item_emb = model.propagate(self._norm_adj)
                u = user_emb[users_t]
                pos = item_emb[pos_t]
                neg = item_emb[neg_t]

                pos_scores = (u * pos).sum(dim=1)
                neg_scores = (u * neg).sum(dim=1)
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model = model

    def _build_norm_adj(self, frame: pd.DataFrame, num_users: int, num_items: int, device: torch.device) -> torch.Tensor:
        user_idx = frame["user_id"].map(self.user_to_idx).to_numpy(dtype=np.int64, copy=False)
        item_idx = frame["item_id"].map(self.item_to_idx).to_numpy(dtype=np.int64, copy=False)
        rows = np.concatenate([user_idx, item_idx + num_users]).astype(np.int64, copy=False)
        cols = np.concatenate([item_idx + num_users, user_idx]).astype(np.int64, copy=False)

        size = int(num_users + num_items)

        # Degree-normalized adjacency for LightGCN uses:
        #   norm_val(u,v) = 1 / sqrt(deg(u) * deg(v))
        # Avoid constructing dense diagonal matrices (which is infeasible for large graphs).
        deg = np.bincount(rows, minlength=size).astype(np.float32, copy=False)
        deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        vals = (deg_inv_sqrt[rows] * deg_inv_sqrt[cols]).astype(np.float32, copy=False)

        idx = torch.tensor(np.stack([rows, cols], axis=0), device=device)
        vals_t = torch.tensor(vals, device=device)
        norm_adj = torch.sparse_coo_tensor(idx, vals_t, (size, size)).coalesce()
        return norm_adj

    def _score_users_items(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        assert self.model is not None and self._norm_adj is not None
        user_emb, item_emb = self.model.propagate(self._norm_adj)
        users = user_emb[user_indices]
        items = item_emb[item_indices]
        scores = users @ items.T
        return scores.mean(dim=0)
