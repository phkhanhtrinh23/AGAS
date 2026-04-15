"""Adapter for the InCarMusic contextual dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from agas.data.loaders.base import DatasetAdapter, clip_chunk
from agas.data.schema import finalize_interactions, finalize_items

_CONTEXT_COLUMNS = [
    "DrivingStyle",
    "landscape",
    "mood",
    "naturalphenomena",
    "RoadType",
    "sleepiness",
    "trafficConditions",
    "weather",
]


class MusicInCarAdapter(DatasetAdapter):
    """Adapter for ``data/music_in_car/Data_InCarMusic.xlsx``."""

    name = "music_in_car"

    def is_available(self, data_root: Path) -> bool:
        """Check whether the InCarMusic workbook exists.

        Args:
            data_root: Root directory containing raw datasets.

        Returns:
            ``True`` when the workbook is present.
        """

        return (data_root / "music_in_car" / "Data_InCarMusic.xlsx").exists()

    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        """Read contextual ratings sheet and yield canonical interaction chunks.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on emitted interaction rows.
            chunk_size: Number of rows yielded per interaction chunk.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        xlsx_path = data_root / "music_in_car" / "Data_InCarMusic.xlsx"
        ratings = pd.read_excel(xlsx_path, sheet_name="ContextualRating")
        ratings.columns = [str(c).strip() for c in ratings.columns]
        ratings = ratings.rename(
            columns={
                "UserID": "user_id",
                "ItemID": "item_id",
                "Rating": "rating",
            }
        )

        def encode_context(row: pd.Series) -> Optional[str]:
            payload = {col: row.get(col) for col in _CONTEXT_COLUMNS if col in row and pd.notna(row.get(col))}
            return json.dumps(payload) if payload else None

        ratings["context_json"] = ratings.apply(encode_context, axis=1)
        ratings["timestamp"] = pd.NA
        ratings["split"] = pd.NA

        rows_left = max_rows
        for start in range(0, len(ratings), chunk_size):
            chunk = ratings.iloc[start : start + chunk_size][
                ["user_id", "item_id", "rating", "timestamp", "split", "context_json"]
            ]
            chunk = clip_chunk(chunk, rows_left)
            if chunk.empty:
                break
            yield finalize_interactions(chunk, dataset=self.name, source="music_in_car/Data_InCarMusic.xlsx")

            if rows_left is not None:
                rows_left -= len(chunk)
                if rows_left <= 0:
                    break

    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load track/category sheets and produce canonical item metadata.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on output item rows.

        Returns:
            Canonical item metadata DataFrame.
        """

        xlsx_path = data_root / "music_in_car" / "Data_InCarMusic.xlsx"
        tracks = pd.read_excel(xlsx_path, sheet_name="Music Track")
        tracks.columns = [str(c).strip() for c in tracks.columns]
        categories = pd.read_excel(xlsx_path, sheet_name="Music Category", header=None, names=["category_id", "category"])

        tracks = tracks.rename(columns={"id": "item_id", "title": "title", "category_id": "category_id"})
        merged = tracks.merge(categories, on="category_id", how="left")
        if max_rows is not None:
            merged = merged.head(max_rows)

        merged["genres"] = merged["category"].fillna("music")
        merged["metadata_json"] = merged.apply(
            lambda row: json.dumps(
                {
                    "artist": row.get("artist"),
                    "album": row.get("album"),
                    "mp3url": row.get("mp3url"),
                }
            ),
            axis=1,
        )
        items = merged[["item_id", "title", "genres", "category", "metadata_json"]]
        return finalize_items(items, dataset=self.name, source="music_in_car/Data_InCarMusic.xlsx")
