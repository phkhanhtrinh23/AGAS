"""Audit estimated LLM worker prompt token usage from a saved episode/transfer JSON.

This script reconstructs the worker LLM prompts that *would* be sent when
`--worker-policy openai|ollama` is used, using the exact prompt templates under
`prompts/` and the episode history to populate per-step context_json.

It does NOT call any LLM provider.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class PromptTokens:
    count: int
    system_tokens_sum: int = 0
    user_tokens_sum: int = 0
    output_tokens_sum_trace: int = 0
    output_tokens_sum_estimated: int = 0
    output_trace_count: int = 0

    @property
    def total_tokens_sum(self) -> int:
        return int(self.system_tokens_sum + self.user_tokens_sum)

    def add(
        self,
        system_tokens: int,
        user_tokens: int,
        *,
        output_tokens_trace: Optional[int],
        output_tokens_estimated: int,
    ) -> None:
        self.count += 1
        self.system_tokens_sum += int(system_tokens)
        self.user_tokens_sum += int(user_tokens)
        self.output_tokens_sum_estimated += int(output_tokens_estimated)
        if output_tokens_trace is not None:
            self.output_trace_count += 1
            self.output_tokens_sum_trace += int(output_tokens_trace)

    def mean(self) -> dict:
        if self.count <= 0:
            return {
                "count": 0,
                "system_mean": 0.0,
                "user_mean": 0.0,
                "total_mean": 0.0,
                "output_trace_mean": None,
                "output_estimated_mean": 0.0,
                "total_io_estimated_mean": 0.0,
                "output_trace_count": 0,
            }
        output_trace_mean = None
        if self.output_trace_count > 0:
            output_trace_mean = float(self.output_tokens_sum_trace / self.output_trace_count)
        return {
            "count": int(self.count),
            "system_mean": float(self.system_tokens_sum / self.count),
            "user_mean": float(self.user_tokens_sum / self.count),
            "total_mean": float(self.total_tokens_sum / self.count),
            "output_trace_mean": output_trace_mean,
            "output_estimated_mean": float(self.output_tokens_sum_estimated / self.count),
            "total_io_estimated_mean": float((self.total_tokens_sum + self.output_tokens_sum_estimated) / self.count),
            "output_trace_count": int(self.output_trace_count),
        }


def _load_history(obj: dict) -> list[dict]:
    for key in ("surrogate_episode_history", "history"):
        hist = obj.get(key)
        if isinstance(hist, list):
            return hist
    raise SystemExit("No episode history found in JSON (expected surrogate_episode_history or history).")


def _agent_trajectory_summary(history: List[dict], agent_id: str, window: int) -> List[Dict[str, Any]]:
    recent = history[-max(1, int(window)) :]
    summary: List[Dict[str, Any]] = []
    for entry in recent:
        step = entry.get("step")
        feedback = entry.get("feedback", {}) or {}
        defense = feedback.get("defense_report") or {}
        reports = entry.get("reports", []) or []
        report = next((r for r in reports if r.get("agent_id") == agent_id), {})
        actions = report.get("actions", []) or []
        outcomes = [o for o in feedback.get("outcomes", []) if (o.get("action") or {}).get("agent_id") == agent_id]
        accepted = [o for o in outcomes if o.get("accepted")]
        discounted = [o for o in outcomes if o.get("discount_applied")]
        dropped = [o for o in outcomes if not o.get("accepted")]
        discount_vals = [abs(o.get("discount_value", 0.0)) for o in discounted]
        mean_discount = sum(discount_vals) / len(discount_vals) if discount_vals else 0.0
        signal = (defense.get("public_signals_by_agent") or {}).get(agent_id, {})
        summary.append(
            {
                "step": int(step),
                "role": report.get("role"),
                "actions": [{"item_id": a.get("item_id"), "rating": a.get("rating")} for a in actions],
                "accepted_actions": len(accepted),
                "dropped_actions": len(dropped),
                "discounted_actions": len(discounted),
                "mean_discount": float(mean_discount),
                "target_rank": int((entry.get("observation") or {}).get("target_rank", 0)),
                "target_rank_delta": int((entry.get("observation") or {}).get("target_rank_delta", 0)),
                "public_signal": signal,
            }
        )
    return summary


def _get_state_before(entry: dict, agent_id: str) -> dict:
    state_before = entry.get("state_before") or {}
    if isinstance(state_before, dict) and agent_id in state_before:
        return state_before[agent_id] or {}
    # Fallback for older outputs that do not store worker snapshots.
    return {}


def _tokenize(enc, text: str) -> int:
    return int(len(enc.encode(text)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-json", required=True, help="Path to outputs/*.json containing episode history.")
    ap.add_argument("--prompt-root", default="prompts")
    ap.add_argument("--trajectory-window", type=int, default=5)
    ap.add_argument("--encoding", default="o200k_base", help="tiktoken encoding name (default: o200k_base).")
    args = ap.parse_args()

    try:
        import tiktoken  # type: ignore
    except Exception as exc:
        raise SystemExit(f"tiktoken is required for this audit: {exc}") from exc

    from agas.agents.messages import AgentRole, RoleAssignment
    from agas.agents.worker import WorkerAgent, WorkerPolicyConfig, WorkerState
    from agas.llm.prompt_store import PromptStore
    from agas.simulation.environment import AGASEnvironment
    from agas.simulation.environment import DefenseConfig
    from agas.agents.defender import DefenseMonitorConfig
    from agas.recsys.surrogate import LightweightSurrogateRecommender, SurrogateConfig
    from agas.recsys.targets import TargetModelConfig
    from agas.cli import _build_episode_recommender, _build_target_config, _load_processed_tables

    obj = json.loads(Path(args.episode_json).read_text(encoding="utf-8"))
    history = _load_history(obj)

    dataset = str(obj.get("dataset", "ml-latest-small"))
    processed_root = Path(str(obj.get("processed_root", "processed")))
    max_interactions_raw = obj.get("max_interactions", None)
    try:
        max_interactions = int(max_interactions_raw) if max_interactions_raw not in (None, "none", "null") else None
    except Exception:
        max_interactions = None

    target_item_id = str(obj.get("target_item_id", "101"))
    target_keyword = str(obj.get("target_keyword", "horror"))
    episode_model = str(obj.get("episode_model", "surrogate")).strip().lower()

    # Rebuild environment to compute the same worker context pools.
    interactions, items = _load_processed_tables(processed_root=processed_root, dataset=dataset, max_interactions=max_interactions)
    dummy_args = argparse.Namespace(
        target_embedding_dim=int(obj.get("target_embedding_dim", 32)),
        target_epochs=int(obj.get("target_epochs", 3)),
        target_batch_size=int(obj.get("target_batch_size", 1024)),
        target_lr=float(obj.get("target_lr", 1e-3)),
        target_weight_decay=float(obj.get("target_weight_decay", 1e-5)),
        target_num_negatives=int(obj.get("target_num_negatives", 4)),
        target_positive_threshold=float(obj.get("target_positive_threshold", 4.0)),
        target_implicit_only=bool(obj.get("target_implicit_only", True)),
        target_explicit_negative_threshold=float(obj.get("target_explicit_negative_threshold", 2.0)),
        seed=int(obj.get("seed", 42)),
        target_device=str(obj.get("target_device", "cpu")),
        target_lightgcn_layers=int(obj.get("target_lightgcn_layers", 2)),
        target_item_vocab=str(obj.get("target_item_vocab", "all")),
        target_item_vocab_max=obj.get("target_item_vocab_max", None),
    )
    target_config: TargetModelConfig = _build_target_config(dummy_args)
    recommender = _build_episode_recommender(
        episode_model,
        interactions,
        items,
        n_factors=int(obj.get("n_factors", 32)),
        target_config=target_config,
    )
    env = AGASEnvironment(
        recommender=recommender,
        base_interactions=interactions,
        items=items,
        target_item_id=target_item_id,
        target_keyword=target_keyword,
        defense_config=DefenseConfig(
            black_box_mode=True,
            monitor_config=DefenseMonitorConfig(),
        ),
        seed=int(obj.get("seed", 42)),
    )
    ctx = env.build_worker_context()

    policy_config = WorkerPolicyConfig(
        implicit_safe=bool(obj.get("implicit_safe_attack", False)),
        positive_threshold=float(obj.get("target_positive_threshold", 4.0)),
        graph_sniper=bool(obj.get("graph_sniper", False)),
        graph_sniper_neighbor_actions=int(obj.get("graph_sniper_neighbor_actions", 3)),
        graph_sniper_include_target=bool(obj.get("graph_sniper_include_target", True)),
        lightgcn_budget=str(obj.get("lightgcn_budget", "small")).lower(),
    )

    store = PromptStore(Path(args.prompt_root))
    enc = tiktoken.get_encoding(str(args.encoding))

    by_role: Dict[str, PromptTokens] = {}
    coordinator_stats = PromptTokens(count=0)

    # Track actions_taken consistently for prompt context (state_before may be absent).
    actions_taken_by_agent: Dict[str, int] = {}

    for step_idx, entry in enumerate(history):
        step = int(entry.get("step", step_idx))

        # Coordinator trace (only present when coordinator-policy is openai/ollama).
        ctrace = entry.get("coordinator_trace")
        if isinstance(ctrace, dict):
            sys_p = ctrace.get("system_prompt")
            usr_p = ctrace.get("user_prompt")
            raw = ctrace.get("raw_response")
            if isinstance(sys_p, str) and isinstance(usr_p, str):
                sys_t = _tokenize(enc, sys_p)
                usr_t = _tokenize(enc, usr_p)
                out_trace = _tokenize(enc, raw) if isinstance(raw, str) and raw.strip() else None
                # If raw response missing, estimate using the assignments dict for that step.
                assignments = entry.get("assignments") if isinstance(entry.get("assignments"), dict) else {}
                role_map = {
                    str(aid): str((assn or {}).get("role", "inactive"))
                    for aid, assn in (assignments or {}).items()
                }
                out_est = json.dumps(role_map, ensure_ascii=False)
                out_est_t = _tokenize(enc, out_est)
                coordinator_stats.add(
                    sys_t,
                    usr_t,
                    output_tokens_trace=out_trace,
                    output_tokens_estimated=out_est_t,
                )
        assignments = entry.get("assignments") or {}
        reports = entry.get("reports") or []
        for report in reports:
            agent_id = str(report.get("agent_id"))
            role = str(report.get("role", "")).strip().lower()
            if role == AgentRole.INACTIVE.value:
                continue

            # State snapshot
            state = _get_state_before(entry, agent_id)
            trust = float(state.get("trust", 0.0))
            risk = float(state.get("risk", 0.0))
            # Prefer tracked counter; fall back to snapshot.
            actions_taken = int(actions_taken_by_agent.get(agent_id, state.get("actions_taken", 0)))

            # Build a WorkerAgent just to reuse its prompt/context logic.
            agent = WorkerAgent(
                state=WorkerState(agent_id=agent_id),
                config=policy_config,
                seed=int(obj.get("seed", 42)) + hash(agent_id) % 10_000,
                llm_client=None,
                prompt_store=store,
                policy_name="openai",
                temperature=0.2,
                total_steps=int(obj.get("num_steps", len(history))),
            )
            agent.state.trust = trust
            agent.state.risk = risk
            agent.state.actions_taken = actions_taken

            # Attach rolling per-agent summary (same structure as episode runner).
            agent_summary = _agent_trajectory_summary(history[:step_idx], agent_id, window=int(args.trajectory_window))
            agent.set_trajectory_summary(agent_summary, total_steps=int(obj.get("num_steps", len(history))))

            meta = dict((assignments.get(agent_id) or {}).get("metadata", {}) or {})
            assn = RoleAssignment(
                step=step,
                agent_id=agent_id,
                role=AgentRole(role),
                rationale=str((assignments.get(agent_id) or {}).get("rationale", "")),
                metadata=meta,
            )

            allowed_items, max_actions = agent._candidate_context(assn.role, ctx, assn)  # type: ignore[attr-defined]
            context = {
                "step": step,
                "agent_id": agent_id,
                "assigned_role": assn.role.value,
                "trust": round(float(trust), 6),
                "risk": round(float(risk), 6),
                "actions_taken": int(actions_taken),
                "target_item_id": str(ctx.target_item_id),
                "allowed_items": allowed_items,
                "max_actions": int(max_actions),
                "benchmark_items": list(ctx.benchmark_items[:20]),
                "target_cluster_items": list(ctx.target_cluster_items[:20]),
                "competitor_items": list(ctx.competitor_items[:10]),
                "noise_items": list(ctx.noise_items[:20]),
                "trajectory_summary": list(agent_summary),
            }

            default_system, default_user = agent._default_prompt_text(assn.role, assn)  # type: ignore[attr-defined]
            bundle = store.load(
                key=f"worker_{assn.role.value}",
                default_system=default_system,
                default_user=default_user,
            )
            context_json = json.dumps(context, indent=2)
            user_prompt = bundle.render_user(
                {
                    "agent_id": agent_id,
                    "step": step,
                    "role": assn.role.value,
                    "max_actions": max_actions,
                    "context_json": context_json,
                }
            )
            system_tokens = _tokenize(enc, bundle.system_prompt)
            user_tokens = _tokenize(enc, user_prompt)

            trace = report.get("trace") if isinstance(report, dict) else None
            raw_response = None
            if isinstance(trace, dict):
                rr = trace.get("raw_response")
                if isinstance(rr, str) and rr.strip():
                    raw_response = rr
            output_tokens_trace = _tokenize(enc, raw_response) if raw_response is not None else None

            # Always provide an estimate by serializing the (already-decided) actions to JSON.
            # This gives a stable, provider-agnostic lower bound for completion size.
            report_actions = report.get("actions") if isinstance(report, dict) else None
            if not isinstance(report_actions, list):
                report_actions = []
            output_estimated = json.dumps({"actions": report_actions}, ensure_ascii=False)
            output_tokens_estimated = _tokenize(enc, output_estimated)

            by_role.setdefault(assn.role.value, PromptTokens(count=0)).add(
                system_tokens,
                user_tokens,
                output_tokens_trace=output_tokens_trace,
                output_tokens_estimated=output_tokens_estimated,
            )

            # Update action counter based on executed actions (in history report).
            actions_taken_by_agent[agent_id] = actions_taken + int(len(report.get("actions") or []))

    out = {role: stats.mean() for role, stats in sorted(by_role.items())}
    out["coordinator"] = coordinator_stats.mean()

    # Aggregate totals (input/output estimated) to make README reporting easier.
    worker_calls = sum(stats.count for stats in by_role.values())
    worker_in_total = sum(stats.total_tokens_sum for stats in by_role.values())
    worker_out_est_total = sum(stats.output_tokens_sum_estimated for stats in by_role.values())

    coord_calls = coordinator_stats.count
    coord_in_total = coordinator_stats.total_tokens_sum
    coord_out_est_total = coordinator_stats.output_tokens_sum_estimated

    out["_totals"] = {
        "steps": int(len(history)),
        "worker_calls": int(worker_calls),
        "worker_input_tokens": int(worker_in_total),
        "worker_output_tokens_estimated": int(worker_out_est_total),
        "worker_total_tokens_estimated": int(worker_in_total + worker_out_est_total),
        "coordinator_calls": int(coord_calls),
        "coordinator_input_tokens": int(coord_in_total),
        "coordinator_output_tokens_estimated": int(coord_out_est_total),
        "coordinator_total_tokens_estimated": int(coord_in_total + coord_out_est_total),
        "total_input_tokens": int(worker_in_total + coord_in_total),
        "total_output_tokens_estimated": int(worker_out_est_total + coord_out_est_total),
        "total_tokens_estimated": int(worker_in_total + worker_out_est_total + coord_in_total + coord_out_est_total),
    }
    out["_meta"] = {
        "episode_json": str(Path(args.episode_json)),
        "dataset": dataset,
        "max_interactions": max_interactions,
        "episode_model": episode_model,
        "target_item_id": target_item_id,
        "target_keyword": target_keyword,
        "encoding": str(args.encoding),
        "trajectory_window": int(args.trajectory_window),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
