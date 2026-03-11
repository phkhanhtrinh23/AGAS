"""Tests for black-box defense signals and prompt-driven worker traces."""

from pathlib import Path

from agas.agents.coordinator import Coordinator, LLMCoordinatorPolicy
from agas.agents.messages import AgentRole, CoordinatorObservation, RatingAction, RoleAssignment, WorkerActionReport
from agas.agents.worker import WorkerAgent, WorkerContext, WorkerState
from agas.data.pipeline import load_interactions, load_items, preprocess_all_datasets
from agas.llm.prompt_store import PromptStore
from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig
from agas.simulation.environment import AGASEnvironment, DefenseConfig


class StaticClient:
    """Minimal test double for deterministic LLM worker responses."""

    def __init__(self, response: str):
        self.response = response
        self.requests = []

    def generate(self, request):  # pragma: no cover - protocol-style shim
        self.requests.append(request)
        return self.response


class RaisingClient:
    """Minimal test double that simulates provider failure."""

    def __init__(self, message: str = "provider failed"):
        self.message = message
        self.requests = []

    def generate(self, request):  # pragma: no cover - protocol-style shim
        self.requests.append(request)
        raise RuntimeError(self.message)


def test_black_box_observation_hides_internal_detection(tmp_path: Path) -> None:
    """Ensure coordinator observations expose only black-box signals, not hidden alerts."""

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

    env = AGASEnvironment(
        recommender=model,
        base_interactions=interactions,
        items=items,
        target_item_id="101",
        target_keyword="horror",
        defense_config=DefenseConfig(black_box_mode=True, spike_threshold=1),
    )
    worker_states = {"agent_1": WorkerState(agent_id="agent_1")}
    report = WorkerActionReport(
        step=0,
        agent_id="agent_1",
        role=AgentRole.SNIPER,
        actions=[RatingAction(agent_id="agent_1", item_id="101", rating=5.0, reason="test target push")],
    )

    feedback = env.execute_step(step=0, reports=[report], worker_states=worker_states)
    observation = env.observation(step=1, worker_states=worker_states)

    assert feedback.defense_report is not None
    assert feedback.defense_report.internal_detection_active is True
    assert observation.alerts_by_agent == {}
    assert observation.lockdown_active is False
    assert observation.signals_by_agent["agent_1"]["attempted_actions"] == 1
    assert observation.signals_by_agent["agent_1"]["suspected_filtering_score"] >= 0.0


def test_llm_worker_loads_prompt_files_and_logs_trace(tmp_path: Path) -> None:
    """Ensure LLM-backed workers load prompts from disk and persist prompt traces."""

    prompt_root = tmp_path / "prompts"
    worker_prompt_dir = prompt_root / "worker_sniper"
    worker_prompt_dir.mkdir(parents=True, exist_ok=True)
    (worker_prompt_dir / "system.txt").write_text("CUSTOM SNIPER SYSTEM", encoding="utf-8")
    (worker_prompt_dir / "user.txt").write_text("CUSTOM USER {{context_json}}", encoding="utf-8")

    client = StaticClient('{"actions":[{"item_id":"101","rating":5.0,"reason":"llm target push"}]}')
    worker = WorkerAgent(
        state=WorkerState(agent_id="agent_1", trust=0.85),
        llm_client=client,
        prompt_store=PromptStore(prompt_root),
        policy_name="openai",
    )
    assignment = RoleAssignment(step=2, agent_id="agent_1", role=AgentRole.SNIPER, rationale="test")
    ctx = WorkerContext(
        target_item_id="101",
        benchmark_items=["1", "2"],
        target_cluster_items=["101", "202"],
        competitor_items=["202"],
        noise_items=["303"],
    )

    report = worker.act(assignment=assignment, step=2, ctx=ctx)

    assert report.policy == "openai"
    assert len(report.actions) == 1
    assert report.actions[0].item_id == "101"
    assert report.actions[0].rating == 5.0
    assert report.trace is not None
    assert report.trace["system_prompt"] == "CUSTOM SNIPER SYSTEM"
    assert report.trace["system_path"] is not None
    assert "CUSTOM USER" in report.trace["user_prompt"]


def test_llm_worker_falls_back_to_rule_actions_when_provider_fails(tmp_path: Path) -> None:
    """Ensure worker execution does not crash when the LLM backend raises an exception."""

    prompt_root = tmp_path / "prompts"
    worker_prompt_dir = prompt_root / "worker_sniper"
    worker_prompt_dir.mkdir(parents=True, exist_ok=True)
    (worker_prompt_dir / "system.txt").write_text("CUSTOM SNIPER SYSTEM", encoding="utf-8")
    (worker_prompt_dir / "user.txt").write_text("CUSTOM USER {{context_json}}", encoding="utf-8")

    client = RaisingClient("openai unavailable")
    worker = WorkerAgent(
        state=WorkerState(agent_id="agent_1"),
        llm_client=client,
        prompt_store=PromptStore(prompt_root),
        policy_name="openai",
    )
    assignment = RoleAssignment(step=0, agent_id="agent_1", role=AgentRole.SNIPER, rationale="test")
    ctx = WorkerContext(
        target_item_id="101",
        benchmark_items=["1", "2"],
        target_cluster_items=["101", "202"],
        competitor_items=["202"],
        noise_items=["303"],
    )

    report = worker.act(assignment=assignment, step=0, ctx=ctx)

    assert len(report.actions) >= 1
    assert report.actions[0].item_id == "101"
    assert report.actions[0].rating == 5.0
    assert report.trace is not None
    assert report.trace["fallback_used"] is True
    assert report.trace["fallback_reason"] == "llm_error"
    assert "openai unavailable" in report.trace["error"]


def test_llm_coordinator_falls_back_to_rule_assignments_when_provider_fails(tmp_path: Path) -> None:
    """Ensure coordinator role assignment falls back cleanly when the LLM backend raises."""

    prompt_root = tmp_path / "prompts"
    coordinator_prompt_dir = prompt_root / "coordinator"
    coordinator_prompt_dir.mkdir(parents=True, exist_ok=True)
    (coordinator_prompt_dir / "system.txt").write_text("COORDINATOR SYSTEM", encoding="utf-8")
    (coordinator_prompt_dir / "user.txt").write_text("COORDINATOR USER {{context_json}}", encoding="utf-8")

    policy = LLMCoordinatorPolicy(
        client=RaisingClient("openai unavailable"),
        agent_order=["agent_1", "agent_2"],
        prompt_store=PromptStore(prompt_root),
    )
    observation = CoordinatorObservation(
        step=0,
        target_item_id="101",
        target_rank=100,
        total_candidates=500,
    )
    worker_states = {
        "agent_1": WorkerState(agent_id="agent_1"),
        "agent_2": WorkerState(agent_id="agent_2"),
    }

    assignments = policy.assign(observation=observation, worker_states=worker_states)

    assert assignments["agent_1"].role == AgentRole.PROFILER
    assert assignments["agent_2"].role == AgentRole.INACTIVE
    assert policy.last_trace is not None
    assert policy.last_trace["fallback_used"] is True
    assert policy.last_trace["fallback_reason"] == "llm_error"
    assert "openai unavailable" in policy.last_trace["error"]


def test_coordinator_injects_periodic_profiler_probe() -> None:
    """Ensure the coordinator does not abandon the profiler role during long runs."""

    class NoProfilerPolicy:
        """Test policy that never assigns profiler on its own."""

        def assign(self, observation, worker_states):  # pragma: no cover - simple test shim
            return {
                "agent_1": RoleAssignment(step=observation.step, agent_id="agent_1", role=AgentRole.SNIPER, rationale="attack"),
                "agent_2": RoleAssignment(step=observation.step, agent_id="agent_2", role=AgentRole.CAMOUFLAGEUR, rationale="cover"),
                "agent_3": RoleAssignment(step=observation.step, agent_id="agent_3", role=AgentRole.CAMOUFLAGEUR, rationale="cover"),
                "agent_4": RoleAssignment(step=observation.step, agent_id="agent_4", role=AgentRole.INACTIVE, rationale="idle"),
            }

    coordinator = Coordinator(policy=NoProfilerPolicy())
    observation = CoordinatorObservation(
        step=6,
        target_item_id="101",
        target_rank=12,
        total_candidates=120,
        target_rank_delta=0,
    )
    worker_states = {
        "agent_1": WorkerState(agent_id="agent_1", trust=1.0, risk=0.6),
        "agent_2": WorkerState(agent_id="agent_2", trust=0.8, risk=0.3),
        "agent_3": WorkerState(agent_id="agent_3", trust=0.7, risk=0.2),
        "agent_4": WorkerState(agent_id="agent_4", trust=0.1, risk=0.0),
    }

    assignments = coordinator.assign_roles(observation=observation, worker_states=worker_states)

    assert any(assignment.role == AgentRole.PROFILER for assignment in assignments.values())


def test_rule_sniper_uses_fixed_max_target_rating() -> None:
    """Ensure the rule sniper always pushes the target with a 5.0 rating."""

    worker = WorkerAgent(state=WorkerState(agent_id="agent_1", trust=0.2, risk=0.1))
    assignment = RoleAssignment(step=2, agent_id="agent_1", role=AgentRole.SNIPER, rationale="test")
    ctx = WorkerContext(
        target_item_id="101",
        benchmark_items=["1", "2"],
        target_cluster_items=["101", "202"],
        competitor_items=["202", "303"],
        noise_items=["404"],
    )

    report = worker.act(assignment=assignment, step=2, ctx=ctx)

    target_actions = [action for action in report.actions if action.item_id == "101"]
    competitor_actions = [action for action in report.actions if action.item_id != "101"]
    assert len(target_actions) == 1
    assert target_actions[0].rating == 5.0
    assert all(action.rating == 1.0 for action in competitor_actions)
