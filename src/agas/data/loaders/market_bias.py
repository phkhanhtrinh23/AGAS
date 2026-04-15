"""Adapter for marketBias datasets (ModCloth + Electronics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from agas.data.loaders.base import DatasetAdapter, clip_chunk
from agas.data.schema import finalize_interactions, finalize_items


class MarketBiasAdapter(DatasetAdapter):
    """Adapter for ``data/marketBias/data`` files."""

    name = "market_bias"

    def is_available(self, data_root: Path) -> bool:
        """Check for both ModCloth and Electronics source CSV files.

        Args:
            data_root: Root directory containing raw datasets.

        Returns:
            ``True`` when both marketBias source files exist.
        """

        folder = data_root / "marketBias" / "data"
        return (folder / "df_modcloth.csv").exists() and (folder / "df_electronics.csv").exists()

    def _iter_subdataset(
        self,
        path: Path,
        domain: str,
        rows_left: Optional[int],
        chunk_size: int,
    ) -> Iterator[pd.DataFrame]:
        """Yield canonical interaction chunks for one marketBias domain file.

        Args:
            path: CSV file path for a specific domain.
            domain: Domain label (for example ``modcloth`` or ``electronics``).
            rows_left: Remaining global interaction budget for this dataset.
            chunk_size: Number of CSV rows to process per iteration.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        for chunk in pd.read_csv(path, chunksize=chunk_size):
            chunk = chunk.rename(columns={"user_id": "raw_user_id", "item_id": "raw_item_id"})
            chunk["user_id"] = chunk["raw_user_id"].map(lambda x: f"{domain}:u:{x}")
            chunk["item_id"] = chunk["raw_item_id"].map(lambda x: f"{domain}:i:{x}")
            chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
            dt = pd.to_datetime(chunk.get("timestamp"), errors="coerce", utc=True)
            chunk["timestamp"] = (dt.astype("int64") // 10**9).astype("Int64")
            chunk["split"] = chunk.get("split")

            context_cols = [c for c in ["category", "brand", "year", "fit", "size", "model_attr", "user_attr"] if c in chunk.columns]
            chunk["context_json"] = chunk[context_cols].apply(
                lambda row: json.dumps({k: v for k, v in row.to_dict().items() if pd.notna(v)}),
                axis=1,
            )

            mapped = chunk[["user_id", "item_id", "rating", "timestamp", "split", "context_json"]]
            mapped = clip_chunk(mapped, rows_left)
            if mapped.empty:
                break
            yield finalize_interactions(mapped, dataset=self.name, source=f"marketBias/data/{path.name}")

            if rows_left is not None:
                rows_left -= len(mapped)
                if rows_left <= 0:
                    break

    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield interactions across domains while preserving a global row budget.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on emitted interactions across both domains.
            chunk_size: Number of rows read per chunk from each domain file.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        folder = data_root / "marketBias" / "data"
        rows_left = max_rows
        for fname, domain in [("df_modcloth.csv", "modcloth"), ("df_electronics.csv", "electronics")]:
            for chunk in self._iter_subdataset(folder / fname, domain, rows_left, chunk_size):
                yield chunk
                if rows_left is not None:
                    rows_left -= len(chunk)
                    if rows_left <= 0:
                        return

    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Build domain-prefixed item metadata table from both marketBias domains.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on output item rows.

        Returns:
            Canonical item metadata DataFrame.
        """

        folder = data_root / "marketBias" / "data"
        frames = []
        for fname, domain in [("df_modcloth.csv", "modcloth"), ("df_electronics.csv", "electronics")]:
            df = pd.read_csv(folder / fname)
            grouped = (
                df.groupby("item_id", as_index=False)
                .agg(
                    avg_rating=("rating", "mean"),
                    category=("category", "first"),
                    brand=("brand", "first"),
                    year=("year", "first"),
                )
                .rename(columns={"item_id": "raw_item_id"})
            )
            grouped["item_id"] = grouped["raw_item_id"].map(lambda x: f"{domain}:i:{x}")
            grouped["title"] = grouped["item_id"].map(lambda x: f"{domain.title()} Item {x.split(':')[-1]}")
            grouped["genres"] = grouped["category"].fillna(domain)
            grouped["category"] = grouped["category"].fillna(domain)
            grouped["metadata_json"] = grouped.apply(
                lambda row: json.dumps(
                    {
                        "domain": domain,
                        "avg_rating": row["avg_rating"],
                        "brand": row["brand"],
                        "year": row["year"],
                    }
                ),
                axis=1,
            )
            frames.append(grouped[["item_id", "title", "genres", "category", "metadata_json"]])

        items = pd.concat(frames, ignore_index=True)
        if max_rows is not None:
            items = items.head(max_rows)
        return finalize_items(items, dataset=self.name, source="marketBias/data")
