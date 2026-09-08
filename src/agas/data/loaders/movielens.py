"""MovieLens dataset adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from agas.data.loaders.base import DatasetAdapter, clip_chunk
from agas.data.schema import finalize_interactions, finalize_items

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


class MovieLensAdapter(DatasetAdapter):
    """Adapter for MovieLens folders with ratings/movies files."""

    def __init__(self, folder_name: str):
        """Bind adapter to a specific MovieLens folder name under ``data/``.

        Args:
            folder_name: MovieLens folder containing ``ratings.csv`` and ``movies.csv``.
        """

        self.folder_name = folder_name
        self.name = folder_name

    def is_available(self, data_root: Path) -> bool:
        """Check whether standard MovieLens ratings/movies CSV files are present.

        Args:
            data_root: Root directory containing dataset folders.

        Returns:
            ``True`` if required MovieLens files exist.
        """

        folder = data_root / self.folder_name
        return (folder / "ratings.csv").exists() and (folder / "movies.csv").exists()

    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        """Stream canonical interaction rows from MovieLens ``ratings.csv``.

        Args:
            data_root: Root directory containing dataset folders.
            max_rows: Optional cap on emitted interaction rows.
            chunk_size: Number of rows read per CSV chunk.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        ratings_path = data_root / self.folder_name / "ratings.csv"
        rows_left = max_rows

        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            mapped = chunk.rename(
                columns={"userId": "user_id", "movieId": "item_id", "rating": "rating", "timestamp": "timestamp"}
            )[["user_id", "item_id", "rating", "timestamp"]]
            mapped["split"] = pd.NA
            mapped["context_json"] = pd.NA
            mapped = clip_chunk(mapped, rows_left)
            if mapped.empty:
                break
            yield finalize_interactions(mapped, dataset=self.name, source=f"{self.folder_name}/ratings.csv")

            if rows_left is not None:
                rows_left -= len(mapped)
                if rows_left <= 0:
                    break

    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load MovieLens movies metadata and enrich with year/primary-genre fields.

        Args:
            data_root: Root directory containing dataset folders.
            max_rows: Optional cap on number of movies loaded.

        Returns:
            Canonical item metadata DataFrame.
        """

        movies_path = data_root / self.folder_name / "movies.csv"
        movies = pd.read_csv(movies_path)
        if max_rows is not None:
            movies = movies.head(max_rows)

        def parse_year(title: str) -> Optional[int]:
            m = _YEAR_RE.search(str(title))
            return int(m.group(1)) if m else None

        movies["primary_genre"] = movies["genres"].fillna("").astype(str).str.split("|").str[0]
        movies["metadata_json"] = movies.apply(
            lambda row: json.dumps(
                {
                    "release_year": parse_year(row.get("title", "")),
                    "primary_genre": row.get("primary_genre"),
                }
            ),
            axis=1,
        )
        items = movies.rename(columns={"movieId": "item_id", "primary_genre": "category"})[
            ["item_id", "title", "genres", "category", "metadata_json"]
        ]
        return finalize_items(items, dataset=self.name, source=f"{self.folder_name}/movies.csv")
