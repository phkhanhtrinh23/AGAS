"""Tests for canonical preprocessing outputs and required schema columns."""

from pathlib import Path

import pandas as pd
import pytest

from agas.data.pipeline import preprocess_all_datasets


@pytest.mark.skipif(
    not Path("data/ml-latest-small").exists(),
    reason="Raw MovieLens dataset not present locally; skip preprocessing integration test.",
)
def test_preprocess_outputs_canonical_files(tmp_path: Path) -> None:
    """Verify preprocessing exports expected datasets and canonical CSV columns."""

    out = tmp_path / "processed"
    stats = preprocess_all_datasets(
        data_root=Path("data"),
        output_root=out,
        include=["ml-latest-small"],
        max_rows_per_dataset=500,
        chunk_size=128,
    )

    datasets = {s.dataset for s in stats}
    assert "ml-latest-small" in datasets

    for s in stats:
        interactions = pd.read_csv(s.output_interactions)
        items = pd.read_csv(s.output_items)
        assert {"dataset", "user_id", "item_id", "rating"}.issubset(interactions.columns)
        assert {"dataset", "item_id", "title"}.issubset(items.columns)
