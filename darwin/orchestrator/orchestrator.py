"""Orchestrator – routes tasks to domain agents and coordinates improvement cycles.

The Orchestrator is the central coordinator of the Darwin self-improvement loop.
It:
1. Polls the Scheduler for pending improvement cycles.
2. Dispatches domain-specific context to each enabled agent.
3. Collects proposals from all agents.
4. Forwards proposals to the Fusion agent.
5. Reports metrics to the Evaluation framework.

Everything runs **locally** – no external APIs, no paid services.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Type

from darwin.agents.art_agent import ArtAgent
from darwin.agents.base import AgentResult, BaseAgent, Proposal
from darwin.agents.code_agent import CodeAgent
from darwin.agents.prompting_agent import PromptingAgent
from darwin.agents.research_agent import ResearchAgent
from darwin.agents.video_agent import VideoAgent
from darwin.core.config import DarwinConfig
from darwin.knowledge.base import KnowledgeBase
from darwin.knowledge.retriever import Retriever
from darwin.orchestrator.scheduler import Scheduler, TaskStatus

logger = logging.getLogger(__name__)

# Registry of all available domain agents
_DOMAIN_AGENT_CLASSES: Dict[str, Type[BaseAgent]] = {
    "code": CodeAgent,
    "art": ArtAgent,
    "video": VideoAgent,
    "prompting": PromptingAgent,
    "research": ResearchAgent,
}


class Orchestrator:
    """Central coordinator for the Darwin self-improvement system.

    Usage::

        cfg = DarwinConfig()
        orch = Orchestrator(cfg)
        orch.start()          # begins the background improvement loop
        # ... system runs autonomously ...
        orch.stop()
    """

    def __init__(
        self,
        cfg: DarwinConfig,
        knowledge_base: Optional[KnowledgeBase] = None,
    ) -> None:
        self.cfg = cfg
        self.scheduler = Scheduler(max_workers=cfg.max_parallel_agents)
        self._agents: Dict[str, BaseAgent] = {}
        self._results_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        # Knowledge Base – primary reference materials supplied by the user.
        # All lookups are local, free, and require no API key.
        self.knowledge_base: KnowledgeBase = knowledge_base or KnowledgeBase(
            store_dir=f"{cfg.data_dir}/knowledge"
        )
        self.retriever = Retriever(self.knowledge_base, top_k=3)

        # Instantiate enabled agents
        for domain in cfg.enabled_domains:
            cls = _DOMAIN_AGENT_CLASSES.get(domain)
            if cls:
                self._agents[domain] = cls()
            else:
                logger.warning("Unknown domain %r – skipping", domain)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler and the background improvement loop."""
        self.scheduler.start()
        self._stop_event.clear()
        self._loop_thread = threading.Thread(
            target=self._improvement_loop, daemon=True, name="darwin-orchestrator"
        )
        self._loop_thread.start()
        logger.info(
            "Orchestrator started with domains: %s",
            list(self._agents.keys()),
        )

    def stop(self) -> None:
        """Gracefully stop the orchestrator."""
        self._stop_event.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=10)
        self.scheduler.stop()
        logger.info("Orchestrator stopped.")

    # ------------------------------------------------------------------
    # Improvement loop
    # ------------------------------------------------------------------

    def _improvement_loop(self) -> None:
        """Continuously run improvement cycles at the configured interval."""
        while not self._stop_event.is_set():
            try:
                self.run_improvement_cycle()
            except Exception as exc:  # noqa: BLE001
                logger.error("Improvement cycle error: %s", exc, exc_info=True)
            # Wait for the next cycle (or until stop is requested)
            self._stop_event.wait(timeout=self.cfg.improvement_interval_seconds)

    def run_improvement_cycle(self, context_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a single improvement cycle synchronously.

        Args:
            context_overrides: Optional per-domain context overrides.

        Returns:
            A summary dict with proposals collected from all agents.
        """
        cycle_start = time.perf_counter()
        logger.info("Starting improvement cycle…")

        all_proposals: List[Proposal] = []
        agent_results: List[AgentResult] = []

        contexts = self._build_agent_contexts(context_overrides)

        if self.scheduler.running:
            # Background mode: dispatch to scheduler workers and collect task results.
            task_ids: Dict[str, str] = {}
            for domain, agent in self._agents.items():
                task_ids[domain] = self.scheduler.submit(
                    fn=agent.run,
                    domain=domain,
                    priority=self._domain_priority(domain),
                    kwargs={"context": contexts[domain]},
                )

            deadline = time.monotonic() + self.cfg.validation_timeout_seconds
            for tid in task_ids.values():
                while time.monotonic() < deadline:
                    status = self.scheduler.get_status(tid)
                    if status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED):
                        break
                    time.sleep(0.05)

            for domain, tid in task_ids.items():
                result = self.scheduler.get_result(tid)
                if isinstance(result, AgentResult):
                    agent_results.append(result)
                else:
                    agent_results.append(
                        AgentResult(
                            agent_id=self._agents[domain].agent_id,
                            domain=domain,
                            error=f"Agent task did not complete: {self.scheduler.get_status(tid)}",
                        )
                    )
        else:
            # One-shot mode used by the CLI/API: run synchronously so a cycle never
            # waits on an inactive scheduler.
            ordered_domains = sorted(self._agents, key=self._domain_priority)
            for domain in ordered_domains:
                agent_results.append(self._agents[domain].run(contexts[domain]))

        for result in agent_results:
            all_proposals.extend(result.proposals)
            if result.error:
                logger.warning("Agent %r error: %s", result.domain, result.error)

        # Sort proposals globally by expected improvement
        all_proposals.sort(key=lambda p: p.expected_improvement, reverse=True)

        duration = time.perf_counter() - cycle_start
        summary = {
            "duration_seconds": round(duration, 2),
            "total_proposals": len(all_proposals),
            "proposals_by_domain": {
                domain: [p.to_dict() for p in all_proposals if p.domain == domain]
                for domain in self._agents
            },
            "agent_errors": {
                r.agent_id: r.error for r in agent_results if r.error
            },
        }

        with self._lock:
            self._results_history.append(summary)

        logger.info(
            "Improvement cycle complete: %d proposals in %.1fs",
            len(all_proposals),
            duration,
        )
        return summary

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _build_context(self, domain: str) -> Dict[str, Any]:
        """Build the context dict passed to each domain agent.

        Automatically enriches the context with the most relevant passages
        from the user's primary reference materials (Knowledge Base).
        """
        base: Dict[str, Any] = {
            "source_root": "darwin",
            "data_dir": self.cfg.data_dir,
        }
        if domain == "code":
            base.update({"focus": ["complexity", "duplication", "coverage", "security", "docstrings"]})
        elif domain == "art":
            base.update({"recent_prompts": [], "quality_scores": {}, "dataset_stats": {}})
        elif domain == "video":
            base.update({"coherence_scores": [], "frame_quality_avg": None, "fps_consistency": 1.0})
        elif domain == "prompting":
            base.update({"prompt_library": [], "response_scores": {}, "domain_prompts": {}})
        elif domain == "research":
            base.update({"corpus_dir": f"{self.cfg.data_dir}/corpus", "qa_benchmark": []})

        # Inject relevant passages from user-supplied reference materials
        if self.knowledge_base.count() > 0:
            query = f"{domain} improvement best practices"
            self.retriever.enrich_agent_context(query, base)

        return base

    def _build_agent_contexts(
        self,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        contexts: Dict[str, Dict[str, Any]] = {}
        for domain in self._agents:
            ctx = self._build_context(domain)
            if context_overrides and domain in context_overrides:
                ctx.update(context_overrides[domain])
            ctx["max_proposals"] = self.cfg.proposal_budget
            contexts[domain] = ctx
        return contexts

    @staticmethod
    def _domain_priority(domain: str) -> int:
        """Lower number = higher scheduler priority."""
        order = {"code": 1, "research": 2, "prompting": 3, "art": 4, "video": 5}
        return order.get(domain, 9)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._results_history)

    def get_agent(self, domain: str) -> Optional[BaseAgent]:
        return self._agents.get(domain)

    def list_domains(self) -> List[str]:
        return list(self._agents.keys())
