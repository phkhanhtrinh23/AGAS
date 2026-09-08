"""Victim-side embedding-space cluster anomaly metric.

Isolated from the AGAS attack pipeline. Requires direct access to the
surrogate model's factor matrices, which are unavailable to attacker agents
in the black-box setting. Use only for post-hoc analysis and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from agas.recsys.surrogate import LightweightSurrogateRecommender


@dataclass
class EmbeddingClusterMetrics:
    """Embedding-space cluster anomaly metrics for the injected fake user pool.

    All cosine values are in [-1, 1]. Higher ``anomaly_score`` means the fake
    user pool is more detectable by an embedding-aware defender.
    """

    n_fake_embedded: int
    """Number of fake workers that received SVD factors (had enough interactions)."""

    n_real_sampled: int
    """Number of real users sampled for the baseline comparison."""

    coverage: float
    """Fraction of workers that were embedded: n_fake_embedded / n_workers."""

    intra_fake_cosine_mean: float
    """Mean pairwise cosine similarity within the fake user pool.

    High (→1) means all fake profiles look alike in latent space —
    a collusion cluster a victim-side analyst could flag.
    """

    intra_fake_cosine_std: float
    """Standard deviation of pairwise intra-fake cosine similarities."""

    fake_real_centroid_cosine_mean: float
    """Mean cosine similarity of each fake user to the real-user centroid.

    High means fake profiles blend well into the real user manifold.
    Low means they are outliers in embedding space.
    """

    separation_ratio: float
    """intra_fake_cosine_mean / fake_real_centroid_cosine_mean.

    >1 means the fake cluster is tighter than the fake-vs-real spread —
    a direct indicator of detectable collusion.
    """

    anomaly_score: float
    """Composite 0-1 detection score: high = attacker is more exposed.

    Combines cohesion (tight fake cluster) and isolation (poor blend with
    real users): ``cohesion_01 * (1 - alignment_01)``.
    """

    per_worker_cosine_distance_to_fake_centroid: Dict[str, float]
    """Per-worker cosine distance to the fake-pool centroid.

    Workers with high distance diverged from the collusion cluster
    (e.g., STEALTH_REBUILD workers that adopted varied cover profiles).
    """

    def to_dict(self) -> dict:
        return {
            "n_fake_embedded": self.n_fake_embedded,
            "n_real_sampled": self.n_real_sampled,
            "coverage": round(self.coverage, 4),
            "intra_fake_cosine_mean": round(self.intra_fake_cosine_mean, 6),
            "intra_fake_cosine_std": round(self.intra_fake_cosine_std, 6),
            "fake_real_centroid_cosine_mean": round(self.fake_real_centroid_cosine_mean, 6),
            "separation_ratio": round(self.separation_ratio, 6),
            "anomaly_score": round(self.anomaly_score, 6),
            "per_worker_cosine_distance_to_fake_centroid": {
                k: round(v, 6) for k, v in self.per_worker_cosine_distance_to_fake_centroid.items()
            },
        }


def _unit_normalize(vecs: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize, guarding against zero vectors."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    return vecs / norms


def compute_embedding_cluster_metrics(
    model: "LightweightSurrogateRecommender",
    worker_ids: Sequence[str],
    real_user_sample_size: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Optional[EmbeddingClusterMetrics]:
    """Compute victim-side embedding cluster anomaly metrics after an episode.

    This function is intentionally decoupled from the AGAS attack loop.
    It requires ``model.user_factors``, which represents internal victim-model
    state and is not accessible to attacker agents in the black-box setting.

    Args:
        model: Fitted LightweightSurrogateRecommender after the episode.
        worker_ids: Fake agent IDs injected during the episode.
        real_user_sample_size: Max real users to sample for comparison baseline.
        rng: Optional RNG for reproducible real-user sampling.

    Returns:
        EmbeddingClusterMetrics, or None when the model has no SVD factors yet
        (too few interactions to factorize).
    """
    # Resolve user embedding matrix — supports MF surrogate (user_factors) and LightGCN.
    user_factors = getattr(model, "user_factors", None)
    if user_factors is None:
        # Try LightGCN: extract propagated user embeddings from the fitted model.
        try:
            import torch
            lgcn_model = getattr(model, "model", None)
            norm_adj = getattr(model, "_norm_adj", None)
            if lgcn_model is not None and norm_adj is not None:
                with torch.no_grad():
                    u_emb, _ = lgcn_model.propagate(norm_adj)
                user_factors = u_emb.cpu().numpy()
            else:
                return None
        except Exception:
            return None

    if rng is None:
        rng = np.random.default_rng(42)

    worker_id_set = {str(w) for w in worker_ids}

    # Fake users that have a row in the embedding matrix.
    embedded_workers = [wid for wid in worker_id_set if wid in model.user_to_idx]
    if not embedded_workers:
        return EmbeddingClusterMetrics(
            n_fake_embedded=0,
            n_real_sampled=0,
            coverage=0.0,
            intra_fake_cosine_mean=0.0,
            intra_fake_cosine_std=0.0,
            fake_real_centroid_cosine_mean=0.0,
            separation_ratio=1.0,
            anomaly_score=0.0,
            per_worker_cosine_distance_to_fake_centroid={},
        )

    fake_indices = np.array([model.user_to_idx[wid] for wid in embedded_workers], dtype=int)
    fake_vecs = user_factors[fake_indices]  # (n_fake, D)
    n_fake = len(embedded_workers)

    # Real user baseline sample
    real_ids = [uid for uid in model.idx_to_user if uid not in worker_id_set]
    if not real_ids:
        return None

    if len(real_ids) > real_user_sample_size:
        chosen = rng.choice(len(real_ids), size=real_user_sample_size, replace=False)
        real_ids_sampled = [real_ids[int(i)] for i in chosen]
    else:
        real_ids_sampled = real_ids

    real_indices = np.array([model.user_to_idx[uid] for uid in real_ids_sampled], dtype=int)
    real_vecs = user_factors[real_indices]  # (n_real, D)

    # Intra-fake pairwise cosine similarity
    if n_fake >= 2:
        fake_normed = _unit_normalize(fake_vecs)
        sim_mat = fake_normed @ fake_normed.T  # (n_fake, n_fake)
        upper_tri = sim_mat[np.triu_indices(n_fake, k=1)]
        intra_fake_mean = float(np.mean(upper_tri))
        intra_fake_std = float(np.std(upper_tri))
    else:
        intra_fake_mean = 1.0  # single point is trivially maximally cohesive
        intra_fake_std = 0.0

    # Real-user centroid and fake-vs-real alignment
    real_centroid = real_vecs.mean(axis=0)
    real_centroid_norm = real_centroid / (np.linalg.norm(real_centroid) + 1e-12)
    fake_normed_for_real = _unit_normalize(fake_vecs)
    fake_to_real_sims = fake_normed_for_real @ real_centroid_norm
    fake_real_centroid_mean = float(np.mean(fake_to_real_sims))

    # Separation ratio
    if abs(fake_real_centroid_mean) < 1e-10:
        separation_ratio = float("inf")
    else:
        separation_ratio = intra_fake_mean / fake_real_centroid_mean

    # Anomaly score: cohesion × isolation, mapped to [0, 1]
    # cohesion_01 = how tight the fake cluster is (higher = more suspicious)
    # alignment_01 = how well fake users blend with real space (higher = less suspicious)
    cohesion_01 = (intra_fake_mean + 1.0) / 2.0
    alignment_01 = (fake_real_centroid_mean + 1.0) / 2.0
    anomaly_score = float(np.clip(cohesion_01 * (1.0 - alignment_01), 0.0, 1.0))

    # Per-worker cosine distance to fake centroid (outlier detection within fake pool)
    fake_centroid = fake_vecs.mean(axis=0)
    fake_centroid_norm = fake_centroid / (np.linalg.norm(fake_centroid) + 1e-12)
    fake_to_centroid_sims = fake_normed_for_real @ fake_centroid_norm
    per_worker = {
        wid: float(1.0 - fake_to_centroid_sims[i])
        for i, wid in enumerate(embedded_workers)
    }

    return EmbeddingClusterMetrics(
        n_fake_embedded=n_fake,
        n_real_sampled=len(real_ids_sampled),
        coverage=n_fake / len(worker_ids),
        intra_fake_cosine_mean=intra_fake_mean,
        intra_fake_cosine_std=intra_fake_std,
        fake_real_centroid_cosine_mean=fake_real_centroid_mean,
        separation_ratio=separation_ratio,
        anomaly_score=anomaly_score,
        per_worker_cosine_distance_to_fake_centroid=per_worker,
    )
