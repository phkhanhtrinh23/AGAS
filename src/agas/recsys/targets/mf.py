"""Matrix factorization style recommenders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTargetRecommender, TargetModelConfig, _build_user_item_sets, _make_positive_pairs


class _DotMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int, nonneg: bool = False):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.nonneg = nonneg

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(users)
        i = self.item_emb(items)
        if self.nonneg:
            u = F.softplus(u)
            i = F.softplus(i)
        return (u * i).sum(dim=1)


def _sample_negatives(
    rng: np.random.Generator,
    user_items: dict[str, set[str]],
    idx_to_item: list[str],
    idx_to_user: list[str],
    num_items: int,
    u_idx: int,
    num_negatives: int,
) -> list[int]:
    user_id = str(idx_to_user[int(u_idx)]) if int(u_idx) < len(idx_to_user) else str(u_idx)
    seen = user_items.get(user_id, set())
    negs: list[int] = []
    for _ in range(num_negatives):
        j = rng.integers(0, num_items)
        while idx_to_item[int(j)] in seen:
            j = rng.integers(0, num_items)
        negs.append(int(j))
    return negs


@dataclass
class MFConfig(TargetModelConfig):
    """MF-specific config placeholder."""


class MFRecommender(BaseTargetRecommender):
    """Matrix factorization with pointwise BCE loss."""

    def __init__(self, config: Optional[MFConfig] = None):
        super().__init__(config=config or MFConfig())
        self.model: Optional[_DotMF] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)

        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _DotMF(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        model = _DotMF(num_users, num_items, self.config.embedding_dim).to(device)
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
        return torch.stack(scores, dim=0).mean(dim=0)


class GMFRecommender(MFRecommender):
    """Generalized matrix factorization (same as MF with BCE in this implementation)."""


class BPRMFRecommender(BaseTargetRecommender):
    """Matrix factorization with BPR loss."""

    def __init__(self, config: Optional[TargetModelConfig] = None):
        super().__init__(config=config or TargetModelConfig())
        self.model: Optional[_DotMF] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)
        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _DotMF(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        model = _DotMF(num_users, num_items, self.config.embedding_dim).to(device)
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
                for u_idx, _ in batch:
                    neg_items.append(
                        _sample_negatives(self._rng, user_items, self.idx_to_item, self.idx_to_user, num_items, int(u_idx), 1)[0]
                    )

                users_t = torch.tensor(users, device=device)
                pos_t = torch.tensor(pos_items, device=device)
                neg_t = torch.tensor(neg_items, device=device)

                pos_scores = model(users_t, pos_t)
                neg_scores = model(users_t, neg_t)
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

                optimizer.zero_grad()
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
        return torch.stack(scores, dim=0).mean(dim=0)


class PMFRecommender(MFRecommender):
    """Probabilistic matrix factorization (approximated with MSE on implicit data)."""

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)

        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _DotMF(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        model = _DotMF(num_users, num_items, self.config.embedding_dim).to(device)
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

                optimizer.zero_grad()
                preds = model(users_t, items_t)
                loss = F.mse_loss(torch.sigmoid(preds), labels_t)
                loss.backward()
                optimizer.step()

        self.model = model


class WMFRecommender(MFRecommender):
    """Weighted MF (implicit) with confidence scaling on positives."""

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)
        alpha = float(getattr(self.config, "wmf_alpha", 1.0))

        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _DotMF(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        model = _DotMF(num_users, num_items, self.config.embedding_dim).to(device)
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
                weights = np.concatenate([
                    np.full(len(users), 1.0 + alpha, dtype=np.float32),
                    np.ones(len(neg_items), dtype=np.float32),
                ])

                users_t = torch.tensor(all_users, device=device)
                items_t = torch.tensor(all_items, device=device)
                labels_t = torch.tensor(labels, device=device)
                weights_t = torch.tensor(weights, device=device)

                optimizer.zero_grad()
                logits = model(users_t, items_t)
                loss = F.binary_cross_entropy_with_logits(logits, labels_t, weight=weights_t)
                loss.backward()
                optimizer.step()

        self.model = model


class NMFRecommender(MFRecommender):
    """Non-negative MF (uses softplus to enforce non-negativity)."""

    def _fit_model(self, frame: pd.DataFrame) -> None:
        assert self.interactions is not None
        device = self._device()
        torch.manual_seed(self.config.seed)

        positives = _make_positive_pairs(frame, self.config.positive_threshold)
        if not positives:
            self.model = _DotMF(len(self.user_to_idx), len(self.item_to_idx), self.config.embedding_dim, nonneg=True).to(device)
            return

        user_items = _build_user_item_sets(frame)
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        model = _DotMF(num_users, num_items, self.config.embedding_dim, nonneg=True).to(device)
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

                optimizer.zero_grad()
                logits = model(users_t, items_t)
                loss = loss_fn(logits, labels_t)
                loss.backward()
                optimizer.step()

        self.model = model
