"""Adapter for MovieLens fractal expansion shards (ml-20mx16x32)."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from agas.data.loaders.base import DatasetAdapter, clip_chunk
from agas.data.schema import finalize_interactions, finalize_items

DEFAULT_ITEM_COUNT = 855_776


class ML20MFractalAdapter(DatasetAdapter):
    """Adapter for ``data/ml-20mx16x32`` npz interaction shards."""

    name = "ml_20mx16x32"

    def is_available(self, data_root: Path) -> bool:
        """Check for both train/test NPZ shard patterns.

        Args:
            data_root: Root directory containing raw datasets.

        Returns:
            ``True`` when expected shard patterns are present.
        """

        folder = data_root / "ml-20mx16x32"
        return any(folder.glob("trainx16x32_*.npz")) and any(folder.glob("testx16x32_*.npz"))

    def _files(self, data_root: Path) -> list[Path]:
        """Return ordered train shards followed by ordered test shards.

        Args:
            data_root: Root directory containing raw datasets.

        Returns:
            Sorted list of NPZ shard file paths.
        """

        folder = data_root / "ml-20mx16x32"
        train = sorted(Path(p) for p in glob.glob(str(folder / "trainx16x32_*.npz")))
        test = sorted(Path(p) for p in glob.glob(str(folder / "testx16x32_*.npz")))
        return train + test

    def iter_interactions(
        self,
        data_root: Path,
        max_rows: Optional[int] = None,
        chunk_size: int = 200_000,
    ) -> Iterator[pd.DataFrame]:
        """Yield implicit-feedback interactions from NPZ arrays in canonical schema.

        Args:
            data_root: Root directory containing raw datasets.
            max_rows: Optional cap on total emitted interactions.
            chunk_size: Number of NPZ rows processed per iteration.

        Returns:
            Iterator of canonical interaction DataFrames.
        """

        emitted = 0

        for file_path in self._files(data_root):
            split = "train" if "train" in file_path.name else "test"
            arr = np.load(file_path)["arr_0"]
            n_rows = arr.shape[0]

            for start in range(0, n_rows, chunk_size):
                if max_rows is not None and emitted >= max_rows:
                    return
                batch = arr[start : start + chunk_size]
                frame = pd.DataFrame(
                    {
                        "user_id": batch[:, 0],
                        "item_id": batch[:, 1],
                        "rating": 1.0,
                        "timestamp": pd.NA,
                        "split": split,
                        "context_json": pd.NA,
                    }
                )
                frame = clip_chunk(frame, None if max_rows is None else (max_rows - emitted))
                if frame.empty:
                    return
                emitted += len(frame)
                yield finalize_interactions(frame, dataset=self.name, source=f"ml-20mx16x32/{file_path.name}")

    def load_items(self, data_root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Synthesize placeholder item metadata for the fractal expansion item space.

        Args:
            data_root: Root directory containing raw datasets (unused).
            max_rows: Optional cap on generated synthetic items.

        Returns:
            Canonical synthetic item metadata DataFrame.
        """

        n_items = DEFAULT_ITEM_COUNT if max_rows is None else min(DEFAULT_ITEM_COUNT, max_rows)
        item_ids = np.arange(n_items)
        items = pd.DataFrame(
            {
                "item_id": item_ids,
                "title": [f"Synthetic Item {i}" for i in item_ids],
                "genres": "synthetic",
                "category": "synthetic",
                "metadata_json": [json.dumps({"synthetic": True}) for _ in item_ids],
            }
        )
        return finalize_items(items, dataset=self.name, source="ml-20mx16x32")
