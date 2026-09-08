"""Adapter for amazon_review dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from agas.data.loaders.base import DatasetAdapter, clip_chunk
from agas.data.schema import finalize_interactions, finalize_items


class AmazonReviewAdapter(DatasetAdapter):
    """Adapter for `data/amazon_review`."""

    name = "amazon_review"

    def is_available(self, data_root: Path) -> bool:
        """Check for required Amazon ratings CSV and review JSONL files.

        Args:
            data_root: Root directory containing raw datasets.

        Returns:
            ``True`` when expected Amazon review files exist.
        """

        folder = data_root / "amazon_review"
        return (folder / "rating-Alabama.csv").exists() and (folder / "review-Alabama_10.json").exists()

    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield canonical interaction chunks from the ratings CSV.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on emitted interaction rows.
            chunk_size: Number of CSV rows read per chunk.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        ratings_path = data_root / "amazon_review" / "rating-Alabama.csv"
        rows_left = max_rows

        for chunk in pd.read_csv(ratings_path, chunksize=chunk_size):
            mapped = chunk.rename(
                columns={"user": "user_id", "business": "item_id", "rating": "rating", "timestamp": "timestamp"}
            )[["user_id", "item_id", "rating", "timestamp"]]
            mapped["timestamp"] = (pd.to_numeric(mapped["timestamp"], errors="coerce") // 1000).astype("Int64")
            mapped["split"] = pd.NA
            mapped["context_json"] = pd.NA
            mapped = clip_chunk(mapped, rows_left)
            if mapped.empty:
                break
            yield finalize_interactions(mapped, dataset=self.name, source="amazon_review/rating-Alabama.csv")

            if rows_left is not None:
                rows_left -= len(mapped)
                if rows_left <= 0:
                    break

    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Aggregate review JSONL into business-level item metadata.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on scanned review rows.

        Returns:
            Canonical item metadata DataFrame.
        """

        review_path = data_root / "amazon_review" / "review-Alabama_10.json"
        counts = defaultdict(int)
        sums = defaultdict(float)

        with review_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                row = json.loads(line)
                item_id = str(row.get("gmap_id"))
                if item_id and item_id != "None":
                    counts[item_id] += 1
                    rating = row.get("rating")
                    if rating is not None:
                        sums[item_id] += float(rating)

        items = pd.DataFrame(
            {
                "item_id": list(counts.keys()),
                "title": [f"Business {x[:12]}" for x in counts.keys()],
                "genres": "local_business",
                "category": "local_business",
                "metadata_json": [
                    json.dumps(
                        {
                            "review_count": counts[k],
                            "avg_rating": (sums[k] / counts[k]) if counts[k] else None,
                        }
                    )
                    for k in counts.keys()
                ],
            }
        )
        return finalize_items(items, dataset=self.name, source="amazon_review/review-Alabama_10.json")
