"""Smoke test for running a short AGAS episode with rule-based coordination."""

from pathlib import Path

from agas.agents.coordinator import RuleBasedCoordinatorPolicy
from agas.data.pipeline import load_interactions, load_items, preprocess_all_datasets
from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig
from agas.simulation.environment import AGASEnvironment
from agas.simulation.episode import AGASEpisodeRunner, EpisodeConfig


def test_episode_runner_smoke(tmp_path: Path) -> None:
    """Validate end-to-end episode execution and basic result shape invariants."""

    processed = tmp_path / "processed"
    preprocess_all_datasets(
        data_root=Path("data"),
        output_root=processed,
        include=["ml-latest-small"],
        max_rows_per_dataset=2000,
        chunk_size=500,
    )

    interactions = load_interactions(processed, "ml-latest-small")
    items = load_items(processed, "ml-latest-small")

    model = LightweightSurrogateRecommender(SurrogateConfig(n_factors=16))
    model.set_items(items)
    model.fit(interactions)

    target_item = "101"
    env = AGASEnvironment(model=model, target_item=target_item, target_genre="Horror")
    runner = AGASEpisodeRunner(
        coordinator=RuleBasedCoordinatorPolicy(),
        environment=env,
        config=EpisodeConfig(n_workers=4, n_steps=3),
    )

    result = runner.run()
    assert len(result.history) == 3
    assert isinstance(result.final_target_rank, int)
    assert result.history[0]["feedback"]["defense_report"] is not None
