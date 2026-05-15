"""Target recommender backends for transfer evaluation.

These backends are torch-dependent. To keep the rest of AGAS importable
without the optional ``targets`` extra, the heavy classes are loaded lazily
on first access. Install them with::

    pip install -e '.[targets]'
"""

from __future__ import annotations

import importlib
from typing import Any

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


_LAZY_IMPORTS = {
    "TargetModelConfig": (".base", "TargetModelConfig"),
    "MFRecommender": (".mf", "MFRecommender"),
    "GMFRecommender": (".mf", "GMFRecommender"),
    "BPRMFRecommender": (".mf", "BPRMFRecommender"),
    "PMFRecommender": (".mf", "PMFRecommender"),
    "WMFRecommender": (".mf", "WMFRecommender"),
    "NMFRecommender": (".mf", "NMFRecommender"),
    "LightGCNRecommender": (".lightgcn", "LightGCNRecommender"),
    "NGCFRecommender": (".ngcf", "NGCFRecommender"),
    "GCMCRecommender": (".gcmc", "GCMCRecommender"),
    "ItemKNNRecommender": (".itemknn", "ItemKNNRecommender"),
    "NeuMFRecommender": (".neumf", "NeuMFRecommender"),
    "SVDppRecommender": (".svdpp", "SVDppRecommender"),
    "IAutoRecRecommender": (".autorec", "IAutoRecRecommender"),
    "UAutoRecRecommender": (".autorec", "UAutoRecRecommender"),
    "CDAERecommender": (".autorec", "CDAERecommender"),
    "SequentialRecommender": (".sequential", "SequentialRecommender"),
}


def __getattr__(name: str) -> Any:
    """Lazily import the requested recommender class on first access.

    This keeps ``import agas.recsys.targets`` cheap when torch is not
    installed; importing or instantiating one of the lazy classes will fail
    with the original :class:`ModuleNotFoundError` from torch.
    """

    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'agas.recsys.targets' has no attribute {name!r}")
    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value
