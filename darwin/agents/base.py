"""Base agent class shared by all domain agents."""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    MERGED = "merged"


@dataclass
class Proposal:
    """A change proposal produced by an agent."""

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = ""
    agent_id: str = ""
    title: str = ""
    description: str = ""
    diff: str = ""                     # unified-diff or structured change
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0            # 0 (safe) – 1 (risky)
    expected_improvement: float = 0.0  # estimated KPI improvement
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validated_at: Optional[datetime] = None
    validation_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "domain": self.domain,
            "agent_id": self.agent_id,
            "title": self.title,
            "description": self.description,
            "diff": self.diff,
            "metadata": self.metadata,
            "risk_score": self.risk_score,
            "expected_improvement": self.expected_improvement,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "validation_notes": self.validation_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Proposal":
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if isinstance(values.get("status"), str):
            values["status"] = ProposalStatus(values["status"])
        if isinstance(values.get("created_at"), str):
            values["created_at"] = datetime.fromisoformat(values["created_at"])
        if isinstance(values.get("validated_at"), str):
            values["validated_at"] = datetime.fromisoformat(values["validated_at"])
        return cls(**values)


@dataclass
class AgentResult:
    """Return value from a single agent run."""

    agent_id: str
    domain: str
    proposals: List[Proposal] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0


class BaseAgent(abc.ABC):
    """Abstract base for all Darwin domain agents.

    Sub-classes must implement:
    * :meth:`analyse` – inspect the current system and produce :class:`Proposal` objects.
    * :meth:`domain` – property returning the domain name (str).
    """

    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.agent_id = agent_id or f"{self.domain}_{uuid.uuid4().hex[:8]}"
        self.config: Dict[str, Any] = config or {}
        self.status = AgentStatus.IDLE
        self._run_history: List[AgentResult] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def domain(self) -> str:
        """Domain name this agent operates on (e.g. 'code', 'art')."""

    @abc.abstractmethod
    def analyse(self, context: Dict[str, Any]) -> List[Proposal]:
        """Analyse *context* and return a list of improvement proposals."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def run(self, context: Dict[str, Any]) -> AgentResult:
        """Execute :meth:`analyse` and record the result."""
        import time

        self.status = AgentStatus.RUNNING
        start = time.perf_counter()
        error: Optional[str] = None
        proposals: List[Proposal] = []
        try:
            proposals = self.analyse(context)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            self.status = AgentStatus.ERROR
        else:
            self.status = AgentStatus.IDLE
        duration = time.perf_counter() - start
        result = AgentResult(
            agent_id=self.agent_id,
            domain=self.domain,
            proposals=proposals,
            error=error,
            duration_seconds=duration,
        )
        self._run_history.append(result)
        return result

    def last_result(self) -> Optional[AgentResult]:
        return self._run_history[-1] if self._run_history else None

    def _make_proposal(
        self,
        title: str,
        description: str,
        diff: str = "",
        risk_score: float = 0.1,
        expected_improvement: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Proposal:
        return Proposal(
            domain=self.domain,
            agent_id=self.agent_id,
            title=title,
            description=description,
            diff=diff,
            risk_score=risk_score,
            expected_improvement=expected_improvement,
            metadata=metadata or {},
        )
