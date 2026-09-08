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

    def select_bridge_items_by_gradient(
        self,
        target_item_id: str,
        candidate_item_ids: list[str],
        n: int = 20,
    ) -> list[str]:
        """Rank candidate items by gradient of target score w.r.t. a soft fake-user edge.

        A cold-start fake user's propagated embedding is:
            u_prop ≈ mean_layers( u_init + sum_j w_j/sqrt(deg_j) * item_prop[j] )
        The gradient d(score(u, target))/d(w_j) selects items whose propagated
        embedding points most strongly toward the target's neighbourhood — these
        create the strongest 2-hop paths without requiring real user histories.

        This is the greedy (single-pass) analogue of GSPAttack's Gumbel-Top-k:
        Gumbel adds end-to-end joint optimisation of all k items; here we use
        independent gradient ranking which is much faster and sufficient for
        episodic profile building.

        Args:
            target_item_id: String item ID of the attack target.
            candidate_item_ids: Pool of items to rank.
            n: Number of top items to return.

        Returns:
            Up to n item IDs ranked by descending gradient magnitude.
        """
        if self.model is None or self._norm_adj is None:
            return candidate_item_ids[:n]

        target_idx = self.item_to_idx.get(str(target_item_id))
        if target_idx is None:
            return candidate_item_ids[:n]

        # Map candidate IDs to model indices; keep only items that actually appear
        # in the training graph (deg > 0).  Zero-degree items have random uninitialised
        # embeddings and no graph connectivity — selecting them would be pure noise.
        num_users = self.model.num_users
        adj_indices = self._norm_adj.coalesce().indices()
        col_ids = adj_indices[1].cpu()
        observed_item_nodes: set[int] = set((col_ids[col_ids >= num_users] - num_users).tolist())

        valid: list[tuple[str, int]] = [
            (item_id, self.item_to_idx[str(item_id)])
            for item_id in candidate_item_ids
            if (
                str(item_id) in self.item_to_idx
                and str(item_id) != str(target_item_id)
                and self.item_to_idx[str(item_id)] in observed_item_nodes
            )
        ]
        if not valid:
            return candidate_item_ids[:n]

        device = next(self.model.parameters()).device

        with torch.no_grad():
            _, item_emb_frozen = self.model.propagate(self._norm_adj)

        # Propagated target embedding (frozen).
        target_emb = item_emb_frozen[target_idx].detach()  # (D,)

        # Soft edge weights over candidate items: w_j ≥ 0, initialised to 1.
        # The fake user's L1-propagated representation is:
        #   u_prop1 = sum_j w_j / sqrt(sum(w) * deg_j) * item_emb[j]
        # Gradient d(u_prop1 · target_emb)/d(w_j) tells which item to add.
        cand_indices = torch.tensor([idx for _, idx in valid], device=device)
        cand_embs = item_emb_frozen[cand_indices].detach()  # (C, D)

        # Degree of each candidate item in current graph.
        # Count nnz per item-node column (cols offset by num_users).
        col_ids = adj_indices[1]  # destination nodes (already on device)
        item_node_ids = cand_indices + num_users  # item nodes in the bipartite graph
        deg_cand = torch.zeros(len(valid), device=device)
        for k, item_node in enumerate(item_node_ids):
            deg_cand[k] = float((col_ids == item_node).sum())
        deg_cand = deg_cand.clamp(min=1.0)

        # Soft weights (all ones = uniform start).
        w = torch.ones(len(valid), device=device, requires_grad=True)

        # u_prop = sum_j (w_j / sqrt(sum(w) * deg_j)) * item_emb[j]
        # To favour well-connected items (higher degree = stronger 2-hop paths)
        # we weight by sqrt(deg_j) instead of the raw normalised form, so the
        # gradient score becomes: sim(item_j, target) * sqrt(deg_j).  This
        # avoids selecting random-embedding, zero-observed items.
        norm_w = w.sum().clamp(min=1e-8)
        scale = w * deg_cand.sqrt() / norm_w             # (C,) — upweight high-deg
        u_prop = (scale.unsqueeze(1) * cand_embs).sum(0)  # (D,)
        score = (u_prop * target_emb).sum()
        score.backward()

        grads = w.grad.detach().cpu().numpy()            # (C,)
        ranked = sorted(zip([item_id for item_id, _ in valid], grads),
                        key=lambda x: -x[1])
        return [item_id for item_id, _ in ranked[:n]]

    def select_bridge_items_by_cooccurrence(
        self,
        target_item_id: str,
        positive_threshold: float = 4.0,
        n: int = 50,
    ) -> list[str]:
        """Select bridge items by co-occurrence with target among segment users.

        Finds items most frequently rated alongside the target by users who
        positively rated the target.  These create the strongest 2-hop paths:
            fake_user → bridge_item → segment_user → target
        without requiring gradient computation.

        Args:
            target_item_id: String item ID of the attack target.
            positive_threshold: Rating threshold to define segment users.
            n: Number of top co-occurring items to return.

        Returns:
            Item IDs sorted by descending co-occurrence count with the target.
        """
        if self.interactions is None:
            return []

        df = self.interactions.copy()
        df["item_id"] = df["item_id"].astype(str)
        df["user_id"] = df["user_id"].astype(str)
        target = str(target_item_id)

        segment_mask = (df["item_id"] == target) & (df["rating"] >= positive_threshold)
        segment_users: set[str] = set(df.loc[segment_mask, "user_id"].tolist())
        if not segment_users:
            segment_users = set(df.loc[df["item_id"] == target, "user_id"].tolist())
        if not segment_users:
            return []

        seg_df = df[df["user_id"].isin(segment_users) & (df["item_id"] != target)]
        cooc = seg_df.groupby("item_id").size().sort_values(ascending=False)
        return cooc.index.astype(str).tolist()[:n]
