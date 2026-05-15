"""Integration smoke test for preprocessing, surrogate fitting, and AGAS loop."""

from pathlib import Path

import pytest

from agas.agents.coordinator import Coordinator, RuleBasedCoordinatorPolicy
from agas.agents.worker import build_worker_pool
from agas.data.pipeline import PreprocessConfig, preprocess_all
from agas.recsys.prepared_data import load_preprocessed_dataset
from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig
from agas.simulation.environment import AGASEnvironment
from agas.simulation.episode import AGASEpisodeRunner, EpisodeConfig, default_agent_ids


@pytest.mark.skipif(
    not Path("data/ml-latest-small").exists(),
    reason="Raw MovieLens dataset not present locally; skip end-to-end smoke test.",
)
def test_episode_smoke(tmp_path: Path) -> None:
    """Check that a 4-step episode completes and returns valid final rank metadata."""

    processed = tmp_path / "processed"
    preprocess_all(
        PreprocessConfig(
            data_root=Path("data"),
            output_root=processed,
            max_rows_per_dataset=20000,
            include_datasets={"ml-latest-small"},
            overwrite=True,
        )
    )

    interactions, items = load_preprocessed_dataset(processed, "ml-latest-small")
    model = LightweightSurrogateRecommender(config=SurrogateConfig(n_factors=16)).fit(interactions)

    agent_ids = default_agent_ids(4)
    workers = build_worker_pool(agent_ids)
    coordinator = Coordinator(policy=RuleBasedCoordinatorPolicy(agent_order=agent_ids))

    env = AGASEnvironment(
        recommender=model,
        base_interactions=interactions,
        items=items,
        target_item_id="101",
        target_keyword="horror",
    )

    runner = AGASEpisodeRunner(
        coordinator=coordinator,
        environment=env,
        workers=workers,
        config=EpisodeConfig(num_steps=4),
    )
    result = runner.run()

    assert len(result.history) == 4
    assert result.final_total_candidates >= 1
    assert result.final_rank >= 1
