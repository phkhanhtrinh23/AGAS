from pathlib import Path

from agas.data.pipeline import PreprocessConfig, preprocess_all


def test_preprocess_smoke(tmp_path: Path) -> None:
    cfg = PreprocessConfig(
        data_root=Path("data"),
        output_root=tmp_path / "processed",
        max_rows_per_dataset=3000,
        include_datasets={"ml-latest-small", "amazon_review", "music_in_car"},
        overwrite=True,
    )
    stats = preprocess_all(cfg)

    datasets = {s.dataset for s in stats}
    assert "ml-latest-small" in datasets
    assert "amazon_review" in datasets
    assert "music_in_car" in datasets

    for s in stats:
        assert s.output_interactions.exists()
        assert s.output_items.exists()
        assert s.interactions > 0

    assert (cfg.output_root / "summary.csv").exists()
