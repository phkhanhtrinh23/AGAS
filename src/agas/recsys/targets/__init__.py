"""Target recommender backends for transfer evaluation."""

from .base import TargetModelConfig
from .lightgcn import LightGCNRecommender
from .neumf import NeuMFRecommender

__all__ = [
    "TargetModelConfig",
    "LightGCNRecommender",
    "NeuMFRecommender",
]
