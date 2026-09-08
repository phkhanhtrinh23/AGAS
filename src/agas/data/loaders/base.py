"""Base abstractions for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


class DatasetAdapter(ABC):
    """Interface for loading one dataset family."""

    name: str

    @abstractmethod
    def is_available(self, data_root: Path) -> bool:
        """Return whether required files for this adapter exist.

        Args:
            data_root: Root dataset directory.

        Returns:
            ``True`` when this adapter can read a dataset from ``data_root``.
        """

        ...

    @abstractmethod
    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield canonical interaction chunks from the raw dataset.

        Args:
            data_root: Root dataset directory.
            max_rows: Optional cap on total emitted interaction rows.
            chunk_size: Preferred chunk size for streaming reads.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        ...

    @abstractmethod
    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Return canonical item metadata table for the dataset.

        Args:
            data_root: Root dataset directory.
            max_rows: Optional cap on number of items returned.

        Returns:
            Canonical item metadata DataFrame.
        """

        ...


def clip_chunk(df: pd.DataFrame, remaining: Optional[int]) -> pd.DataFrame:
    """Clip rows by remaining quota.

    Args:
        df: Input DataFrame chunk.
        remaining: Remaining number of rows allowed, or ``None`` for no limit.

    Returns:
        Original DataFrame or a row-limited view according to ``remaining``.
    """

    if remaining is None:
        return df
    if remaining <= 0:
        return df.iloc[0:0]
    return df.iloc[:remaining]
