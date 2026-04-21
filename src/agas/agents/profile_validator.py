"""Profile validator: detect fake-looking profiles from action history.

Computes a per-agent suspicion score combining standard shilling-attack
signatures — rating extremity, target lock-in, and cross-agent co-voting —
into an aggregate in ``[0, 1]``. The score is injected into the coordinator
observation so rule-based and LLM policies can read it and force cool-down
for agents whose score crosses a threshold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class _AgentRecord:
    ratings: List[Tuple[str, float]] = field(default_factory=list)


class ProfileValidator:
    """Accumulates per-agent rating history and scores profile authenticity."""

    def __init__(
        self,
        *,
        extremity_weight: float = 0.4,
        target_hit_weight: float = 0.3,
        collusion_weight: float = 0.3,
        positive_threshold: float = 4.0,
    ) -> None:
        """Initialize metric weights.

        Args:
            extremity_weight: Weight on fraction of ratings at extreme values.
            target_hit_weight: Weight on positive-rated target signal.
            collusion_weight: Weight on cross-agent co-voting overlap.
            positive_threshold: Rating threshold counted as "positive" for target-hit.
        """

        self._records: Dict[str, _AgentRecord] = defaultdict(_AgentRecord)
        self._extremity_w = float(extremity_weight)
        self._target_hit_w = float(target_hit_weight)
        self._collusion_w = float(collusion_weight)
        self._positive_threshold = float(positive_threshold)

    def update(self, agent_id: str, actions) -> None:
        """Append a step of ``(item_id, rating)`` pairs for one agent."""

        rec = self._records[str(agent_id)]
        for action in actions:
            item_id = getattr(action, "item_id", None)
            rating = getattr(action, "rating", None)
            if item_id is None or rating is None:
                continue
            rec.ratings.append((str(item_id), float(rating)))

    def _extremity(self, ratings: List[Tuple[str, float]]) -> float:
        """Fraction of ratings at extreme values (1.0 or 5.0)."""

        if not ratings:
            return 0.0
        extreme = sum(1 for _, r in ratings if r >= 4.5 or r <= 1.5)
        return extreme / len(ratings)

    def _target_hit(self, ratings: List[Tuple[str, float]], target_item_id: str) -> float:
        """1.0 if any rating on the target is positive, else 0.0."""

        target = str(target_item_id)
        for item_id, r in ratings:
            if item_id == target and r >= self._positive_threshold:
                return 1.0
        return 0.0

    def _collusion(self, agent_id: str) -> float:
        """Fraction of agent's items also rated by >=2 other agents in our pool.

        Captures coordinated bridge-item inflation typical of AGAS snipers
        +profilers flocking to the same co-occurrence bridges.
        """

        own_items = {item for item, _ in self._records[agent_id].ratings}
        if not own_items:
            return 0.0
        item_counts: Dict[str, int] = defaultdict(int)
        for aid, rec in self._records.items():
            if aid == agent_id:
                continue
            for item, _ in rec.ratings:
                item_counts[item] += 1
        overlap = sum(1 for it in own_items if item_counts.get(it, 0) >= 2)
        return overlap / len(own_items)

    def score_all(self, target_item_id: str) -> Dict[str, Dict[str, float]]:
        """Return per-agent metric breakdown and aggregate score."""

        out: Dict[str, Dict[str, float]] = {}
        for agent_id, rec in self._records.items():
            extremity = self._extremity(rec.ratings)
            target_hit = self._target_hit(rec.ratings, target_item_id)
            collusion = self._collusion(agent_id)
            aggregate = (
                self._extremity_w * extremity
                + self._target_hit_w * target_hit
                + self._collusion_w * collusion
            )
            out[agent_id] = {
                "extremity": round(extremity, 4),
                "target_hit": round(target_hit, 4),
                "collusion": round(collusion, 4),
                "aggregate": round(min(1.0, aggregate), 4),
                "n_ratings": float(len(rec.ratings)),
            }
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the validator state."""

        return {
            aid: {
                "n_ratings": len(rec.ratings),
                "ratings": list(rec.ratings),
            }
            for aid, rec in self._records.items()
        }
