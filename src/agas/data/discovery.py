"""Dataset adapter registry and discovery.

The public benchmark in ``experiment.tex`` uses exactly six datasets:

* MovieLens-100K (ML-100K)
* MovieLens-1M (ML-1M)
* MovieLens Tag Genome 2021 (Genome 2021)
* Netflix Prize (Netflix)
* Douban Movie (Douban)
* Amazon Reviews 2018 (Amazon)

The default adapter registry therefore only enables loaders for those six
families plus the tiny ``ml-latest-small`` sample used by the smoke tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from agas.data.loaders.amazon_review import AmazonReviewAdapter
from agas.data.loaders.base import DatasetAdapter
from agas.data.loaders.genome2021 import Genome2021Adapter
from agas.data.loaders.movielens import MovieLensAdapter
from agas.data.loaders.netflix import NetflixAdapter


def build_default_adapters() -> List[DatasetAdapter]:
    """Return adapters for the six datasets used by the paper, plus the
    ``ml-latest-small`` sample used by smoke tests."""

    return [
        MovieLensAdapter("ml-latest-small"),
        MovieLensAdapter("ml-100k"),
        MovieLensAdapter("ml-1m"),
        Genome2021Adapter(),
        NetflixAdapter(),
        AmazonReviewAdapter(),
    ]


def discover_available_adapters(
    data_root: Path, adapters: Iterable[DatasetAdapter] | None = None
) -> List[DatasetAdapter]:
    """Return adapters whose required files exist in ``data_root``.

    Args:
        data_root: Root directory where raw datasets are stored.
        adapters: Optional adapter list to filter; when ``None`` uses defaults.

    Returns:
        Adapters that report themselves available under ``data_root``.
    """

    candidates = list(adapters) if adapters is not None else build_default_adapters()
    return [adapter for adapter in candidates if adapter.is_available(data_root)]
