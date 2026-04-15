"""Dataset adapter registry and discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from agas.data.loaders.amazon_review import AmazonReviewAdapter
from agas.data.loaders.base import DatasetAdapter
from agas.data.loaders.fake_review import FakeReviewAdapter
from agas.data.loaders.genome2021 import Genome2021Adapter
from agas.data.loaders.market_bias import MarketBiasAdapter
from agas.data.loaders.ml20m_fractal import ML20MFractalAdapter
from agas.data.loaders.movielens import MovieLensAdapter
from agas.data.loaders.music_in_car import MusicInCarAdapter
from agas.data.loaders.netflix import NetflixAdapter


def build_default_adapters() -> List[DatasetAdapter]:
    """Return all adapters supported by this repository."""

    return [
        MovieLensAdapter("ml-latest-small"),
        MovieLensAdapter("ml-latest"),
        MovieLensAdapter("ml-32m"),
        ML20MFractalAdapter(),
        Genome2021Adapter(),
        NetflixAdapter(),
        AmazonReviewAdapter(),
        FakeReviewAdapter(),
        MarketBiasAdapter(),
        MusicInCarAdapter(),
    ]


def discover_available_adapters(data_root: Path, adapters: Iterable[DatasetAdapter] | None = None) -> List[DatasetAdapter]:
    """Return adapters whose required files exist in ``data_root``.

    Args:
        data_root: Root directory where raw datasets are stored.
        adapters: Optional adapter list to filter; when ``None`` uses defaults.

    Returns:
        Adapters that report themselves available under ``data_root``.
    """

    candidates = list(adapters) if adapters is not None else build_default_adapters()
    return [adapter for adapter in candidates if adapter.is_available(data_root)]
