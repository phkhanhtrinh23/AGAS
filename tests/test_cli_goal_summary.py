"""Tests for CLI-level goal-rank summary fields."""

from agas.cli import _summarize_goal_status


def test_goal_summary_reports_initial_and_best_rank_progress() -> None:
    """Ensure goal summary captures best rank and first goal-reaching step."""

    history = [
        {"step": 0, "feedback": {"target_rank": 120}},
        {"step": 1, "feedback": {"target_rank": 40}},
        {"step": 2, "feedback": {"target_rank": 4}},
    ]

    summary = _summarize_goal_status(initial_rank=150, history=history, goal_rank=5)

    assert summary["goal_rank"] == 5
    assert summary["best_rank"] == 4
    assert summary["best_rank_step"] == 2
    assert summary["goal_achieved"] is True
    assert summary["goal_first_reached_step"] == 2


def test_goal_summary_marks_failure_when_goal_is_never_reached() -> None:
    """Ensure failed episodes are marked clearly when no step reaches the goal rank."""

    history = [
        {"step": 0, "feedback": {"target_rank": 120}},
        {"step": 1, "feedback": {"target_rank": 90}},
    ]

    summary = _summarize_goal_status(initial_rank=150, history=history, goal_rank=5)

    assert summary["best_rank"] == 90
    assert summary["best_rank_step"] == 1
    assert summary["goal_achieved"] is False
    assert summary["goal_first_reached_step"] is None
