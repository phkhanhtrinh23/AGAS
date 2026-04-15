"""Netflix Prize dataset adapter.

Reads the "combined_data_1..4.txt" training files where each movie block looks like:

  MovieID:
  CustomerID,Rating,Date
  CustomerID,Rating,Date
  ...

and `movie_titles.csv` containing:

  MovieID,YearOfRelease,Title
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from agas.data.loaders.base import DatasetAdapter, clip_chunk
from agas.data.schema import finalize_interactions, finalize_items


_EPOCH = date(1970, 1, 1)


def _date_to_unix_days(iso_date: str) -> int:
    """Convert YYYY-MM-DD -> integer days since 1970-01-01."""

    # Fast path parsing without datetime.strptime overhead.
    # Expected format is always 10 chars: YYYY-MM-DD.
    year = int(iso_date[0:4])
    month = int(iso_date[5:7])
    day = int(iso_date[8:10])
    return (date(year, month, day) - _EPOCH).days


@dataclass(frozen=True)
class NetflixAdapter(DatasetAdapter):
    """Adapter for Netflix Prize `data/netflix/` exports."""

    name: str = "netflix"

    def is_available(self, data_root: Path) -> bool:
        folder = data_root / "netflix"
        return (folder / "movie_titles.csv").exists() and (folder / "combined_data_1.txt").exists()

    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        folder = data_root / "netflix"
        sources = [
            ("netflix/combined_data_1.txt", folder / "combined_data_1.txt"),
            ("netflix/combined_data_2.txt", folder / "combined_data_2.txt"),
            ("netflix/combined_data_3.txt", folder / "combined_data_3.txt"),
            ("netflix/combined_data_4.txt", folder / "combined_data_4.txt"),
        ]
        rows_left = max_rows

        for source_name, path in sources:
            if not path.exists():
                continue

            user_ids: list[str] = []
            item_ids: list[str] = []
            ratings: list[int] = []
            timestamps: list[int] = []

            cur_movie: Optional[str] = None

            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue

                    if line.endswith(":"):
                        cur_movie = line[:-1]
                        continue

                    if cur_movie is None:
                        continue

                    # CustomerID,Rating,Date
                    parts = line.split(",")
                    if len(parts) < 3:
                        continue
                    user_id = parts[0].strip()
                    rating = parts[1].strip()
                    d = parts[2].strip()
                    if not user_id or not rating or len(d) < 10:
                        continue

                    user_ids.append(user_id)
                    item_ids.append(cur_movie)
                    try:
                        ratings.append(int(rating))
                    except ValueError:
                        # Skip malformed rating rows.
                        user_ids.pop()
                        item_ids.pop()
                        continue
                    try:
                        timestamps.append(_date_to_unix_days(d))
                    except Exception:
                        # Skip malformed dates.
                        user_ids.pop()
                        item_ids.pop()
                        ratings.pop()
                        continue

                    if len(user_ids) >= chunk_size:
                        chunk = pd.DataFrame(
                            {
                                "user_id": user_ids,
                                "item_id": item_ids,
                                "rating": ratings,
                                "timestamp": timestamps,
                                "split": pd.NA,
                                "context_json": pd.NA,
                            }
                        )
                        chunk = clip_chunk(chunk, rows_left)
                        if chunk.empty:
                            return
                        yield finalize_interactions(chunk, dataset=self.name, source=source_name)
                        if rows_left is not None:
                            rows_left -= int(len(chunk))
                            if rows_left <= 0:
                                return
                        user_ids, item_ids, ratings, timestamps = [], [], [], []

            if user_ids:
                chunk = pd.DataFrame(
                    {
                        "user_id": user_ids,
                        "item_id": item_ids,
                        "rating": ratings,
                        "timestamp": timestamps,
                        "split": pd.NA,
                        "context_json": pd.NA,
                    }
                )
                chunk = clip_chunk(chunk, rows_left)
                if chunk.empty:
                    return
                yield finalize_interactions(chunk, dataset=self.name, source=source_name)
                if rows_left is not None:
                    rows_left -= int(len(chunk))
                    if rows_left <= 0:
                        return

    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        titles_path = data_root / "netflix" / "movie_titles.csv"
        movies = pd.read_csv(
            titles_path,
            header=None,
            names=["item_id", "year", "title"],
            encoding="latin-1",
            on_bad_lines="skip",
        )
        if max_rows is not None:
            movies = movies.head(max_rows)

        # Netflix Prize does not include genres. To keep the rest of the AGAS
        # pipeline working (cluster selection via --target-keyword), we store the
        # title text in the `genres` field so keyword matching still produces a
        # non-empty cluster.
        movies["genres"] = movies["title"].fillna("").astype(str)
        movies["category"] = "movie"
        movies["metadata_json"] = movies["year"].apply(lambda y: json.dumps({"release_year": int(y)}) if pd.notna(y) else pd.NA)

        items = movies[["item_id", "title", "genres", "category", "metadata_json"]]
        return finalize_items(items, dataset=self.name, source="netflix/movie_titles.csv")

