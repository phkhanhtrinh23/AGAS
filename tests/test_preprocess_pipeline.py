"""Smoke tests for full preprocessing pipeline execution and artifact creation."""

from pathlib import Path

import pytest

from agas.data.pipeline import PreprocessConfig, preprocess_all


@pytest.mark.skipif(
    not Path("data/ml-latest-small").exists(),
    reason="Raw MovieLens dataset not present locally; skip preprocessing integration test.",
)
def test_preprocess_smoke(tmp_path: Path) -> None:
    """Ensure selected datasets preprocess successfully and produce summary artifacts."""

    cfg = PreprocessConfig(
        data_root=Path("data"),
        output_root=tmp_path / "processed",
        max_rows_per_dataset=3000,
        include_datasets={"ml-latest-small"},
        overwrite=True,
    )
    stats = preprocess_all(cfg)

    datasets = {s.dataset for s in stats}
    assert "ml-latest-small" in datasets

    for s in stats:
        assert s.output_interactions.exists()
        assert s.output_items.exists()
        assert s.interactions > 0

    assert (cfg.output_root / "summary.csv").exists()
