"""Message protocol for coordinator-worker-environment communication."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentRole(str, Enum):
    """Supported worker roles in AGAS."""

    PROFILER = "profiler"
    CAMOUFLAGEUR = "camouflaguer"
    SNIPER = "sniper"
    INACTIVE = "inactive"


@dataclass
class ProtocolMessage:
    """Generic transport envelope used by all components."""

    sender: str
    receiver: str
    step: int
    message_type: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RatingAction:
    """Single rating action emitted by one worker."""

    agent_id: str
    item_id: str
    rating: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoordinatorObservation:
    """Environment state observed by coordinator each step."""

    step: int
    target_item_id: str
    target_rank: int
    total_candidates: int
    alerts_by_agent: Dict[str, str] = field(default_factory=dict)
    trust_by_agent: Dict[str, float] = field(default_factory=dict)
    risk_by_agent: Dict[str, float] = field(default_factory=dict)
    lockdown_active: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoleAssignment:
    """Coordinator decision assigning a role to one worker."""

    step: int
    agent_id: str
    role: AgentRole
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        return data


@dataclass
class WorkerActionReport:
    """Actions prepared by a worker for current step."""

    step: int
    agent_id: str
    role: AgentRole
    actions: List[RatingAction] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "agent_id": self.agent_id,
            "role": self.role.value,
            "actions": [a.to_dict() for a in self.actions],
            "notes": self.notes,
        }


@dataclass
class ActionOutcome:
    """Result of applying one action in environment."""

    action: RatingAction
    accepted: bool
    effective_rating: Optional[float]
    trust_delta: float
    risk_delta: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "accepted": self.accepted,
            "effective_rating": self.effective_rating,
            "trust_delta": self.trust_delta,
            "risk_delta": self.risk_delta,
            "reason": self.reason,
        }


@dataclass
class EnvironmentFeedback:
    """Environment response after executing one step."""

    step: int
    outcomes: List[ActionOutcome]
    alerts_by_agent: Dict[str, str]
    target_rank: int
    total_candidates: int
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "alerts_by_agent": dict(self.alerts_by_agent),
            "target_rank": self.target_rank,
            "total_candidates": self.total_candidates,
            "notes": self.notes,
        }
