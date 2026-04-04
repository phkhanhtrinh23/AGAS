"""AutoRec and CDAE recommenders (simplified)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTargetRecommender, TargetModelConfig


class _AutoRec(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.encoder(x))
        return self.decoder(h)


class _CDAE(nn.Module):
    def __init__(self, num_users: int, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.dropout(x)) + self.user_emb(users)
        h = F.relu(h)
        return self.decoder(h)


@dataclass
class AutoRecConfig(TargetModelConfig):
    """AutoRec config placeholder."""


class IAutoRecRecommender(BaseTargetRecommender):
    """Item-based AutoRec."""

    def __init__(self, config: Optional[AutoRecConfig] = None):
        super().__init__(config=config or AutoRecConfig())
        self.model: Optional[_AutoRec] = None
        self._item_matrix: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        device = self._device()
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        hidden_dim = int(getattr(self.config, "autorec_hidden_dim", max(16, self.config.embedding_dim)))

        mat = np.zeros((num_items, num_users), dtype=np.float32)
        for row in frame.itertuples(index=False):
            u = self.user_to_idx.get(str(row.user_id))
            i = self.item_to_idx.get(str(row.item_id))
            if u is None or i is None:
                continue
            mat[i, u] = float(row.rating)
        mask = (mat > 0).astype(np.float32)

        model = _AutoRec(num_users, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        mat_t = torch.tensor(mat, device=device)
        mask_t = torch.tensor(mask, device=device)
        batch_size = max(1, int(self.config.batch_size))
        for _ in range(int(self.config.epochs)):
            perm = torch.randperm(num_items, device=device)
            for start in range(0, num_items, batch_size):
                idx = perm[start : start + batch_size]
                x = mat_t[idx]
                m = mask_t[idx]
                optimizer.zero_grad()
                recon = model(x)
                loss = ((recon - x) * m).pow(2).sum() / (m.sum() + 1e-8)
                loss.backward()
                optimizer.step()

        self.model = model
        self._item_matrix = mat_t
        self._mask = mask_t

    def _score_users_items(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        assert self.model is not None and self._item_matrix is not None
        device = self._device()
        items = item_indices.to(device)
        x = self._item_matrix[items]
        recon = self.model(x)
        scores = recon[:, user_indices.to(device)].mean(dim=1)
        return scores


class UAutoRecRecommender(BaseTargetRecommender):
    """User-based AutoRec."""

    def __init__(self, config: Optional[AutoRecConfig] = None):
        super().__init__(config=config or AutoRecConfig())
        self.model: Optional[_AutoRec] = None
        self._user_matrix: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        device = self._device()
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        hidden_dim = int(getattr(self.config, "autorec_hidden_dim", max(16, self.config.embedding_dim)))

        mat = np.zeros((num_users, num_items), dtype=np.float32)
        for row in frame.itertuples(index=False):
            u = self.user_to_idx.get(str(row.user_id))
            i = self.item_to_idx.get(str(row.item_id))
            if u is None or i is None:
                continue
            mat[u, i] = float(row.rating)
        mask = (mat > 0).astype(np.float32)

        model = _AutoRec(num_items, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        mat_t = torch.tensor(mat, device=device)
        mask_t = torch.tensor(mask, device=device)
        batch_size = max(1, int(self.config.batch_size))
        for _ in range(int(self.config.epochs)):
            perm = torch.randperm(num_users, device=device)
            for start in range(0, num_users, batch_size):
                idx = perm[start : start + batch_size]
                x = mat_t[idx]
                m = mask_t[idx]
                optimizer.zero_grad()
                recon = model(x)
                loss = ((recon - x) * m).pow(2).sum() / (m.sum() + 1e-8)
                loss.backward()
                optimizer.step()

        self.model = model
        self._user_matrix = mat_t
        self._mask = mask_t

    def _score_users_items(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        assert self.model is not None and self._user_matrix is not None
        device = self._device()
        users = user_indices.to(device)
        items = item_indices.to(device)
        x = self._user_matrix[users]
        recon = self.model(x)
        scores = recon[:, items].mean(dim=0)
        return scores


class CDAERecommender(BaseTargetRecommender):
    """Collaborative Denoising Auto-Encoder (simplified)."""

    def __init__(self, config: Optional[AutoRecConfig] = None):
        super().__init__(config=config or AutoRecConfig())
        self.model: Optional[_CDAE] = None
        self._user_matrix: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None

    def _fit_model(self, frame: pd.DataFrame) -> None:
        device = self._device()
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        hidden_dim = int(getattr(self.config, "autorec_hidden_dim", max(16, self.config.embedding_dim)))
        dropout = float(getattr(self.config, "cdae_dropout", 0.5))

        mat = np.zeros((num_users, num_items), dtype=np.float32)
        for row in frame.itertuples(index=False):
            u = self.user_to_idx.get(str(row.user_id))
            i = self.item_to_idx.get(str(row.item_id))
            if u is None or i is None:
                continue
            mat[u, i] = float(row.rating)
        mask = (mat > 0).astype(np.float32)

        model = _CDAE(num_users, num_items, hidden_dim, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        mat_t = torch.tensor(mat, device=device)
        mask_t = torch.tensor(mask, device=device)
        batch_size = max(1, int(self.config.batch_size))
        for _ in range(int(self.config.epochs)):
            perm = torch.randperm(num_users, device=device)
            for start in range(0, num_users, batch_size):
                idx = perm[start : start + batch_size]
                x = mat_t[idx]
                m = mask_t[idx]
                users_t = idx
                optimizer.zero_grad()
                recon = model(x, users_t)
                loss = ((recon - x) * m).pow(2).sum() / (m.sum() + 1e-8)
                loss.backward()
                optimizer.step()

        self.model = model
        self._user_matrix = mat_t
        self._mask = mask_t

    def _score_users_items(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        assert self.model is not None and self._user_matrix is not None
        device = self._device()
        users = user_indices.to(device)
        items = item_indices.to(device)
        x = self._user_matrix[users]
        recon = self.model(x, users)
        scores = recon[:, items].mean(dim=0)
        return scores
