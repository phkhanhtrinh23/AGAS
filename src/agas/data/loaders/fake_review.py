"""Adapter for fake-review-detection dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from agas.data.loaders.base import DatasetAdapter, clip_chunk
from agas.data.schema import finalize_interactions, finalize_items


class FakeReviewAdapter(DatasetAdapter):
    """Adapter for YelpZip-derived fake review dataset."""

    name = "fake_review_detection"

    def is_available(self, data_root: Path) -> bool:
        """Check for required train/test TSV files in the fake-review folder.

        Args:
            data_root: Root directory containing raw datasets.

        Returns:
            ``True`` when expected fake-review files exist.
        """

        folder = data_root / "fake-review-detection"
        return (folder / "new_data_train.csv").exists() and (folder / "new_data_test.csv").exists()

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        """Read dataset file and strip unnamed artifact columns.

        Args:
            path: Input TSV-like file path.

        Returns:
            Parsed DataFrame without unnamed artifact columns.
        """

        df = pd.read_csv(path, sep="\t", engine="python")
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or c == ""]
        if unnamed:
            df = df.drop(columns=unnamed)
        return df

    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield canonical interactions across train/test files with contextual payload.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on total emitted interactions.
            chunk_size: Number of rows emitted per batch from each split.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        folder = data_root / "fake-review-detection"
        rows_left = max_rows

        for split_name, filename in [("train", "new_data_train.csv"), ("test", "new_data_test.csv")]:
            df = self._read(folder / filename)
            for start in range(0, len(df), chunk_size):
                chunk = df.iloc[start : start + chunk_size].copy()
                chunk = chunk.rename(
                    columns={
                        "reviewerID": "user_id",
                        "restaurantID": "item_id",
                        "rating": "rating",
                        "date": "date",
                        "flagged": "flagged",
                        "reviewUsefulCount": "review_useful_count",
                        "restaurantRating": "restaurant_rating",
                    }
                )
                dt = pd.to_datetime(chunk["date"], errors="coerce")
                chunk["timestamp"] = (dt.astype("int64") // 10**9).astype("Int64")
                chunk["split"] = split_name
                chunk["context_json"] = chunk.apply(
                    lambda row: json.dumps(
                        {
                            "flagged": row.get("flagged"),
                            "review_useful_count": row.get("review_useful_count"),
                            "restaurant_rating": row.get("restaurant_rating"),
                        }
                    ),
                    axis=1,
                )
                out = chunk[["user_id", "item_id", "rating", "timestamp", "split", "context_json"]]
                out = clip_chunk(out, rows_left)
                if out.empty:
                    return
                yield finalize_interactions(out, dataset=self.name, source=f"fake-review-detection/{filename}")

                if rows_left is not None:
                    rows_left -= len(out)
                    if rows_left <= 0:
                        return

    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Aggregate restaurant-level metadata from combined train/test files.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on source rows used for aggregation.

        Returns:
            Canonical item metadata DataFrame.
        """

        folder = data_root / "fake-review-detection"
        train_df = self._read(folder / "new_data_train.csv")
        test_df = self._read(folder / "new_data_test.csv")
        df = pd.concat([train_df, test_df], ignore_index=True)
        if max_rows is not None:
            df = df.head(max_rows)

        grouped = (
            df.groupby("restaurantID", as_index=False)
            .agg(
                title=("restaurantID", "first"),
                avg_rating=("restaurantRating", "mean"),
                n_reviews=("reviewID", "count"),
            )
            .rename(columns={"restaurantID": "item_id"})
        )
        grouped["category"] = "restaurant"
        grouped["genres"] = "restaurant"
        grouped["metadata_json"] = grouped.apply(
            lambda row: json.dumps(
                {
                    "avg_restaurant_rating": row.get("avg_rating"),
                    "n_reviews": int(row.get("n_reviews", 0)),
                }
            ),
            axis=1,
        )
        items = grouped[["item_id", "title", "genres", "category", "metadata_json"]]
        return finalize_items(items, dataset=self.name, source="fake-review-detection/new_data_*.csv")
