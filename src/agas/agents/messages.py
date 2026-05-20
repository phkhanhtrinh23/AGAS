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


class VictimModelClass(str, Enum):
    """Inferred architecture class of the victim recommender system.

    Used by the coordinator to choose the optimal attack strategy and by workers
    to select the right sniper variant.

    UNKNOWN       – not yet classified (probe phase in progress or skipped).
    MF_STYLE      – matrix-factorisation / MLP models (NeuMF, GMF, BPR-MF).
                    Direct target rating gives a clean positive gradient → more
                    is better.  Fake new users preferred.
    LIGHTGCN_STYLE – degree-normalised graph models (LightGCN, NGCF, SimGCL).
                    Direct target rating inflates the item's degree and dilutes ALL
                    existing edges.  Must attack indirectly via cluster neighbours
                    and competitor degree inflation.
    SEQUENTIAL_STYLE – recency-sensitive sequence models (SASRec, GRU4Rec, BERT4Rec).
                    The position of the target rating in the interaction sequence
                    matters: rating the target *last* (after genre-consistent fillers)
                    exploits the next-item prediction bias.
    """

    UNKNOWN = "unknown"
    MF_STYLE = "mf_style"
    LIGHTGCN_STYLE = "lightgcn_style"
    SEQUENTIAL_STYLE = "sequential_style"


@dataclass
class ProtocolMessage:
    """Generic transport envelope used by all components."""

    sender: str
    receiver: str
    step: int
    message_type: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass payload into a plain JSON-serializable dict.

        Returns:
            Dictionary representation of the protocol message.
        """

        return asdict(self)


@dataclass
class RatingAction:
    """Single rating action emitted by one worker."""

    agent_id: str
    item_id: str
    rating: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation for logging and transport.

        Returns:
            Dictionary representation of the rating action.
        """

        return asdict(self)


@dataclass
class CoordinatorObservation:
    """Environment state observed by coordinator each step."""

    step: int
    target_item_id: str
    target_rank: int
    total_candidates: int
    target_rank_delta: int = 0
    total_steps: int | None = None
    trajectory_summary: Optional[List[Dict[str, Any]]] = None
    agent_memory_by_agent: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    alerts_by_agent: Dict[str, str] = field(default_factory=dict)
    trust_by_agent: Dict[str, float] = field(default_factory=dict)
    risk_by_agent: Dict[str, float] = field(default_factory=dict)
    signals_by_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lockdown_active: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full observation for policy input or trace output.

        Returns:
            Dictionary representation of the coordinator observation.
        """

        return asdict(self)


@dataclass
class RoleAssignment:
    """Coordinator decision assigning a role to one worker."""

    step: int
    agent_id: str
    role: AgentRole
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize assignment while converting enum roles to wire-format strings.

        Returns:
            Dictionary representation of the role assignment.
        """

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
    policy: str = "openai"
    trace: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize worker actions and metadata for history capture.

        Returns:
            Dictionary representation of the worker report.
        """

        return {
            "step": self.step,
            "agent_id": self.agent_id,
            "role": self.role.value,
            "actions": [a.to_dict() for a in self.actions],
            "notes": self.notes,
            "policy": self.policy,
            "trace": self.trace,
        }


@dataclass
class ActionOutcome:
    """Result of applying one action in environment."""

    action: RatingAction
    accepted: bool
    effective_rating: Optional[float]
    discount_applied: bool
    discount_value: float
    trust_delta: float
    risk_delta: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize environment decision for one rating action.

        Returns:
            Dictionary representation of the action outcome.
        """

        return {
            "action": self.action.to_dict(),
            "accepted": self.accepted,
            "effective_rating": self.effective_rating,
            "discount_applied": self.discount_applied,
            "discount_value": self.discount_value,
            "trust_delta": self.trust_delta,
            "risk_delta": self.risk_delta,
            "reason": self.reason,
        }


@dataclass
class AgentBlackBoxSignal:
    """Indirect, black-box observable signal for one worker."""

    agent_id: str
    attempted_actions: int = 0
    accepted_actions: int = 0
    dropped_actions: int = 0
    discounted_actions: int = 0
    acceptance_rate: float = 1.0
    discount_rate: float = 0.0
    mean_discount: float = 0.0
    suppression_streak: int = 0
    suspected_filtering_score: float = 0.0
    group_overlap: float = 0.0
    group_suspicion: float = 0.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize one worker's black-box response signal.

        Returns:
            Dictionary representation of the black-box signal.
        """

        return asdict(self)


@dataclass
class DefenseReport:
    """Defense monitor output with hidden ground truth and public signals."""

    step: int
    internal_detection_active: bool
    hidden_lockdown_active: bool
    internal_detection_by_agent: Dict[str, List[str]] = field(default_factory=dict)
    public_signals_by_agent: Dict[str, AgentBlackBoxSignal] = field(default_factory=dict)
    target_rank_before: int = 0
    target_rank_after: int = 0
    target_rank_delta: int = 0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize defense monitor report for logging and analysis.

        Returns:
            Dictionary representation of the defense report.
        """

        return {
            "step": self.step,
            "internal_detection_active": self.internal_detection_active,
            "hidden_lockdown_active": self.hidden_lockdown_active,
            "internal_detection_by_agent": {
                aid: list(reasons) for aid, reasons in self.internal_detection_by_agent.items()
            },
            "public_signals_by_agent": {
                aid: signal.to_dict() for aid, signal in self.public_signals_by_agent.items()
            },
            "target_rank_before": self.target_rank_before,
            "target_rank_after": self.target_rank_after,
            "target_rank_delta": self.target_rank_delta,
            "notes": self.notes,
        }


@dataclass
class EnvironmentFeedback:
    """Environment response after executing one step."""

    step: int
    outcomes: List[ActionOutcome]
    alerts_by_agent: Dict[str, str]
    target_rank: int
    total_candidates: int
    defense_report: Optional[DefenseReport] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full per-step environment feedback for the episode timeline.

        Returns:
            Dictionary representation of the environment feedback.
        """

        return {
            "step": self.step,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "alerts_by_agent": dict(self.alerts_by_agent),
            "target_rank": self.target_rank,
            "total_candidates": self.total_candidates,
            "defense_report": self.defense_report.to_dict() if self.defense_report is not None else None,
            "notes": self.notes,
        }
