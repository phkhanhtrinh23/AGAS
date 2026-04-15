"""Adapter for Tag Genome 2021 raw ratings dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd

from agas.data.loaders.base import DatasetAdapter, clip_chunk
from agas.data.schema import finalize_interactions, finalize_items


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    """Yield decoded JSON objects from one JSONL file, skipping blank lines.

    Args:
        path: JSONL file path.

    Returns:
        Iterator of decoded JSON objects (dicts).
    """

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class Genome2021Adapter(DatasetAdapter):
    """Adapter for ``data/genome_2021/raw``."""

    name = "genome_2021"

    def is_available(self, data_root: Path) -> bool:
        """Check for required raw ratings and metadata JSONL files.

        Args:
            data_root: Root directory containing raw datasets.

        Returns:
            ``True`` when expected Genome 2021 files exist.
        """

        raw = data_root / "genome_2021" / "raw"
        return (raw / "ratings.json").exists() and (raw / "metadata.json").exists()

    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        """Stream ratings JSONL into canonical interaction chunks.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on emitted interaction rows.
            chunk_size: Number of buffered JSON rows per yielded chunk.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        ratings_path = data_root / "genome_2021" / "raw" / "ratings.json"

        rows = []
        emitted = 0
        for row in _iter_jsonl(ratings_path):
            rows.append(
                {
                    "user_id": row.get("user_id"),
                    "item_id": row.get("item_id"),
                    "rating": row.get("rating"),
                    "timestamp": pd.NA,
                    "split": pd.NA,
                    "context_json": pd.NA,
                }
            )
            if len(rows) >= chunk_size:
                chunk = pd.DataFrame(rows)
                rows = []
                chunk = clip_chunk(chunk, None if max_rows is None else (max_rows - emitted))
                if chunk.empty:
                    break
                emitted += len(chunk)
                yield finalize_interactions(chunk, dataset=self.name, source="genome_2021/raw/ratings.json")
                if max_rows is not None and emitted >= max_rows:
                    return

        if rows and (max_rows is None or emitted < max_rows):
            chunk = pd.DataFrame(rows)
            chunk = clip_chunk(chunk, None if max_rows is None else (max_rows - emitted))
            if not chunk.empty:
                yield finalize_interactions(chunk, dataset=self.name, source="genome_2021/raw/ratings.json")

    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load movie metadata JSONL and project selected fields into item records.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on emitted item rows.

        Returns:
            Canonical item metadata DataFrame.
        """

        metadata_path = data_root / "genome_2021" / "raw" / "metadata.json"
        rows = []
        for idx, row in enumerate(_iter_jsonl(metadata_path)):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append(
                {
                    "item_id": row.get("item_id"),
                    "title": row.get("title"),
                    "genres": pd.NA,
                    "category": "movie",
                    "metadata_json": json.dumps(
                        {
                            "directedBy": row.get("directedBy"),
                            "starring": row.get("starring"),
                            "avgRating": row.get("avgRating"),
                            "imdbId": row.get("imdbId"),
                        }
                    ),
                }
            )

        items = pd.DataFrame(rows)
        return finalize_items(items, dataset=self.name, source="genome_2021/raw/metadata.json")
