"""Target recommender backends for transfer evaluation."""

from .autorec import CDAERecommender, IAutoRecRecommender, UAutoRecRecommender
from .base import TargetModelConfig
from .gcmc import GCMCRecommender
from .itemknn import ItemKNNRecommender
from .lightgcn import LightGCNRecommender
from .mf import (
    BPRMFRecommender,
    GMFRecommender,
    MFRecommender,
    NMFRecommender,
    PMFRecommender,
    WMFRecommender,
)
from .neumf import NeuMFRecommender
from .ngcf import NGCFRecommender
from .sequential import SequentialRecommender
from .svdpp import SVDppRecommender

__all__ = [
    "TargetModelConfig",
    "MFRecommender",
    "GMFRecommender",
    "BPRMFRecommender",
    "PMFRecommender",
    "WMFRecommender",
    "NMFRecommender",
    "LightGCNRecommender",
    "NGCFRecommender",
    "GCMCRecommender",
    "ItemKNNRecommender",
    "NeuMFRecommender",
    "SVDppRecommender",
    "IAutoRecRecommender",
    "UAutoRecRecommender",
    "CDAERecommender",
    "SequentialRecommender",
]
