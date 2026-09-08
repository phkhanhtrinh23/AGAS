"""Smoke test for running a short AGAS episode with rule-based coordination."""

from pathlib import Path

import pytest

from agas.agents.coordinator import RuleBasedCoordinatorPolicy
from agas.data.pipeline import load_interactions, load_items, preprocess_all_datasets
from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig
from agas.simulation.environment import AGASEnvironment
from agas.simulation.episode import AGASEpisodeRunner, EpisodeConfig

pytestmark = pytest.mark.skipif(
    not Path("data/ml-latest-small").exists(),
    reason="Raw MovieLens dataset not present locally; skip episode integration tests.",
)


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
        config=EpisodeConfig(n_workers=4, n_steps=3, stop_on_goal=False),
    )

    result = runner.run()
    assert len(result.history) == 3
    assert isinstance(result.final_target_rank, int)
    assert result.history[0]["feedback"]["defense_report"] is not None


def test_episode_runner_stops_immediately_when_goal_already_reached(tmp_path: Path) -> None:
    """Ensure the runner exits without executing steps when the initial rank already meets the goal."""

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

    env = AGASEnvironment(model=model, target_item="101", target_genre="Horror")
    runner = AGASEpisodeRunner(
        coordinator=RuleBasedCoordinatorPolicy(),
        environment=env,
        config=EpisodeConfig(n_workers=4, n_steps=3, goal_rank=env.current_rank, stop_on_goal=True),
    )

    result = runner.run()
    assert result.executed_steps == 0
    assert result.stopped_early is True
    assert result.stop_reason is not None


def test_episode_runner_can_disable_stop_on_goal(tmp_path: Path) -> None:
    """Ensure the runner continues executing when early stopping is explicitly disabled."""

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

    env = AGASEnvironment(model=model, target_item="101", target_genre="Horror")
    runner = AGASEpisodeRunner(
        coordinator=RuleBasedCoordinatorPolicy(),
        environment=env,
        config=EpisodeConfig(n_workers=4, n_steps=2, goal_rank=env.current_rank, stop_on_goal=False),
    )

    result = runner.run()
    assert result.executed_steps == 2
    assert result.stopped_early is False
