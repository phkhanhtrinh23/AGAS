"""Lightweight surrogate recommender used by AGAS simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


@dataclass
class SurrogateConfig:
    """Hyperparameters for the lightweight surrogate recommender."""

    n_factors: int = 32
    reg_lambda: float = 1.0
    random_state: int = 42
    min_interactions_for_factors: int = 1_000


class LightweightSurrogateRecommender:
    """Matrix-factorization surrogate with popularity fallback."""

    def __init__(self, config: Optional[SurrogateConfig] = None):
        """Initialize empty model state and optional training configuration.

        Args:
            config: Optional surrogate hyperparameter configuration.
        """

        self.config = config or SurrogateConfig()
        self.interactions: pd.DataFrame | None = None
        self.items: pd.DataFrame | None = None

        self.user_to_idx: dict[str, int] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_user: list[str] = []
        self.idx_to_item: list[str] = []

        self.global_mean: float = 0.0
        self.item_bias: np.ndarray | None = None
        self.item_popularity: np.ndarray | None = None

        self._matrix: csr_matrix | None = None
        self._svd: TruncatedSVD | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def set_items(self, items: pd.DataFrame) -> None:
        """Attach canonical item metadata used by downstream simulation components.

        Args:
            items: Canonical items DataFrame.
        """

        self.items = items.copy()

    def fit(self, interactions: pd.DataFrame) -> "LightweightSurrogateRecommender":
        """Train the surrogate from canonical interactions.

        Args:
            interactions: Canonical interactions DataFrame containing at least
                ``user_id``, ``item_id``, and ``rating``.

        Returns:
            ``self`` after fitting.

        Training steps:
        1. Validate and normalize key columns.
        2. Build a sparse user-item matrix.
        3. Compute global/item priors (mean, bias, popularity).
        4. Optionally fit TruncatedSVD factors when enough interactions exist.
        """

        required = {"user_id", "item_id", "rating"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"Missing required interaction columns: {missing}")

        frame = interactions[["user_id", "item_id", "rating"]].copy()
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)
        frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
        frame = frame.dropna(subset=["user_id", "item_id", "rating"])
        frame = frame.groupby(["user_id", "item_id"], as_index=False)["rating"].mean()
        self.interactions = frame

        self.idx_to_user = sorted(frame["user_id"].unique().tolist())
        self.idx_to_item = sorted(frame["item_id"].unique().tolist())
        # print(f"### idx_to_user: {self.idx_to_user[:20]} ...")
        self.user_to_idx = {u: i for i, u in enumerate(self.idx_to_user)}
        # print(f"### user_to_idx: {list(self.user_to_idx.keys())[:20]} ...")
        self.item_to_idx = {i: j for j, i in enumerate(self.idx_to_item)}
        # print(f"### item_to_idx: {list(self.item_to_idx.keys())[:20]} ...")
        user_idx = frame["user_id"].map(self.user_to_idx).to_numpy()
        # print(f"### user_idx: {user_idx}")

        item_idx = frame["item_id"].map(self.item_to_idx).to_numpy()
        # print(f"### item_idx: {item_idx}")

        values = frame["rating"].astype(float).to_numpy()
        # print(f"### values: {values}")

        n_users = len(self.idx_to_user)
        n_items = len(self.idx_to_item)
        self._matrix = csr_matrix((values, (user_idx, item_idx)), shape=(n_users, n_items))
        # print(f"### matrix shape: {self._matrix.shape}, nnz: {self._matrix.nnz}")

        self.global_mean = float(values.mean()) if len(values) else 0.0
        item_group = frame.groupby("item_id")["rating"]
        bias_series = item_group.mean()
        pop_series = item_group.count()

        self.item_bias = np.array([float(bias_series.get(i, self.global_mean)) for i in self.idx_to_item])
        self.item_popularity = np.array([float(pop_series.get(i, 0.0)) for i in self.idx_to_item])

        # print(f"### global_mean: {self.global_mean}")
        # print(f"### item_bias: {self.item_bias[:20]} ...")
        # print(f"### can_factorize: {len(values)} interactions, {n_users} users, {n_items} items")
        # print(f"### min_interactions_for_factors: {self.config.min_interactions_for_factors}")
        can_factorize = len(values) >= self.config.min_interactions_for_factors and min(n_users, n_items) >= 3
        if can_factorize:
            n_components = min(self.config.n_factors, min(n_users, n_items) - 1)
            self._svd = TruncatedSVD(n_components=n_components, random_state=self.config.random_state)
            self.user_factors = self._svd.fit_transform(self._matrix)
            # print(f"### user_factors shape: {self.user_factors.shape}")
            self.item_factors = self._svd.components_.T
        else:
            self._svd = None
            self.user_factors = None
            self.item_factors = None

        return self

    def append_interactions(self, new_interactions: pd.DataFrame, refit: bool = True) -> None:
        """Append new rows and optionally refit model state to simulate online updates.

        Args:
            new_interactions: New interaction rows with canonical columns.
            refit: Whether to immediately refit model statistics/factors.
        """

        if self.interactions is None:
            self.fit(new_interactions)
            return

        appended = pd.concat([self.interactions, new_interactions[["user_id", "item_id", "rating"]]], ignore_index=True)
        if refit:
            self.fit(appended)
        else:
            self.interactions = appended

    def _infer_user_vector(self, profile: Dict[str, float]) -> Optional[np.ndarray]:
        """Infer a latent user vector from explicit ratings via regularized least squares.

        Args:
            profile: Mapping ``item_id -> rating`` for the target user profile.

        Returns:
            Inferred latent vector, or ``None`` when factors/items are unavailable.
        """

        if self.item_factors is None:
            return None

        item_indices = [self.item_to_idx[i] for i in profile.keys() if i in self.item_to_idx]
        if not item_indices:
            return None

        X = self.item_factors[item_indices, :]
        y = np.array([float(profile[self.idx_to_item[idx]]) for idx in item_indices], dtype=float)

        reg = self.config.reg_lambda * np.eye(X.shape[1], dtype=float)
        lhs = X.T @ X + reg
        rhs = X.T @ y
        return np.linalg.solve(lhs, rhs)

    def _scores_from_vector(self, user_vector: Optional[np.ndarray]) -> np.ndarray:
        """Compute item scores from priors plus optional latent personalization signal.

        Args:
            user_vector: Optional latent user vector for personalization.

        Returns:
            Score array aligned with ``idx_to_item`` ordering.
        """

        if self.item_bias is None:
            raise RuntimeError("Model must be fit before scoring")

        base = self.item_bias.copy()
        if user_vector is not None and self.item_factors is not None:
            base = base + self.item_factors @ user_vector

        if self.item_popularity is not None:
            pop = np.log1p(self.item_popularity)
            pop = pop / (pop.max() if pop.max() > 0 else 1.0)
            base = base + 0.05 * pop
        return base

    def recommend(
        self,
        user_id: Optional[str] = None,
        profile: Optional[Dict[str, float]] = None,
        top_n: int = 10,
        candidate_items: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Return top-N unseen item IDs for a user ID and/or explicit profile hints.

        Args:
            user_id: Optional known user ID already present in fitted interactions.
            profile: Optional explicit ``item_id -> rating`` profile hints.
            top_n: Maximum number of recommendations to return.
            candidate_items: Optional item subset to rank instead of all items.

        Returns:
            Ranked item ID list excluding items already seen by the user/profile.
        """

        if self.interactions is None:
            raise RuntimeError("Model is not fit")

        seen_items: set[str] = set()
        user_vector: Optional[np.ndarray] = None

        if user_id is not None and user_id in self.user_to_idx and self.user_factors is not None:
            idx = self.user_to_idx[user_id]
            user_vector = self.user_factors[idx]
            seen = self.interactions.loc[self.interactions["user_id"] == user_id, "item_id"]
            seen_items.update(seen.astype(str).tolist())

        if profile:
            seen_items.update(map(str, profile.keys()))
            inferred = self._infer_user_vector({str(k): float(v) for k, v in profile.items()})
            if inferred is not None:
                user_vector = inferred if user_vector is None else 0.5 * (user_vector + inferred)

        scores = self._scores_from_vector(user_vector)

        if candidate_items is None:
            candidate_idx = np.arange(len(self.idx_to_item))
        else:
            candidate_idx = np.array([self.item_to_idx[i] for i in candidate_items if i in self.item_to_idx], dtype=int)

        if candidate_idx.size == 0:
            return []

        candidate_scores = scores[candidate_idx]
        order = candidate_idx[np.argsort(-candidate_scores)]

        out: list[str] = []
        for idx in order:
            item_id = self.idx_to_item[int(idx)]
            if item_id in seen_items:
                continue
            out.append(item_id)
            if len(out) >= top_n:
                break
        return out

    def mean_scores_for_segment(
        self,
        segment_user_ids: Optional[Iterable[str]] = None,
        candidate_items: Optional[Sequence[str]] = None,
    ) -> dict[str, float]:
        """Estimate mean item scores for a user segment over a candidate set.

        Args:
            segment_user_ids: Optional user subset used for segment averaging.
            candidate_items: Optional item subset to score.

        Returns:
            Mapping ``item_id -> mean score``.
        """

        if self.interactions is None:
            raise RuntimeError("Model is not fit")

        if candidate_items is None:
            candidate_indices = np.arange(len(self.idx_to_item))
        else:
            candidate_indices = np.array([self.item_to_idx[i] for i in candidate_items if i in self.item_to_idx], dtype=int)

        if candidate_indices.size == 0:
            return {}

        if self.user_factors is None or self.item_factors is None:
            base = self._scores_from_vector(None)
            return {self.idx_to_item[i]: float(base[i]) for i in candidate_indices}

        if segment_user_ids:
            user_indices = [self.user_to_idx[u] for u in segment_user_ids if u in self.user_to_idx]
            if not user_indices:
                user_indices = list(range(len(self.idx_to_user)))
        else:
            user_indices = list(range(len(self.idx_to_user)))

        user_matrix = self.user_factors[np.array(user_indices)]
        item_matrix = self.item_factors[candidate_indices]
        pred = user_matrix @ item_matrix.T
        mean_pred = pred.mean(axis=0)

        base = self._scores_from_vector(None)
        final = mean_pred + base[candidate_indices]

        return {self.idx_to_item[int(item_idx)]: float(score) for item_idx, score in zip(candidate_indices, final)}

    def rank_item(
        self,
        item_id: str,
        segment_user_ids: Optional[Iterable[str]] = None,
        candidate_items: Optional[Sequence[str]] = None,
    ) -> tuple[int, int]:
        """Return 1-based rank of ``item_id`` and total candidate count.

        Args:
            item_id: Target item ID to rank.
            segment_user_ids: Optional user subset used for segment averaging.
            candidate_items: Optional candidate item subset for ranking.

        Returns:
            Tuple ``(rank, total_candidates)`` with 1-based rank.
        """

        scores = self.mean_scores_for_segment(segment_user_ids=segment_user_ids, candidate_items=candidate_items)
        if not scores:
            return (0, 0)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for pos, (cur_item, _) in enumerate(ordered, start=1):
            if cur_item == item_id:
                return (pos, len(ordered))
        return (len(ordered) + 1, len(ordered))
