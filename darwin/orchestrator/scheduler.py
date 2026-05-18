"""Task scheduler for the Darwin multi-agent system.

Uses a simple priority queue.  No external broker needed – everything runs
in-process (or via Python's multiprocessing for true parallelism).
All components are free and open-source.
"""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class ScheduledTask:
    """A unit of work with priority and scheduling metadata."""

    priority: int                         # lower = higher priority
    scheduled_at: float = field(compare=True)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    domain: str = field(default="", compare=False)
    fn: Callable[..., Any] = field(default=lambda: None, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    status: TaskStatus = field(default=TaskStatus.QUEUED, compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[str] = field(default=None, compare=False)
    started_at: Optional[float] = field(default=None, compare=False)
    finished_at: Optional[float] = field(default=None, compare=False)


class Scheduler:
    """Thread-safe priority scheduler.

    Tasks are stored in a min-heap keyed by (priority, scheduled_at).
    A background thread pops and executes tasks in order.

    Example::

        scheduler = Scheduler(max_workers=2)
        scheduler.start()
        scheduler.submit(my_fn, domain="code", priority=1, kwargs={"x": 1})
        scheduler.stop()
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._heap: List[ScheduledTask] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._max_workers = max_workers
        self._workers: List[threading.Thread] = []
        self._running = False
        self._task_registry: Dict[str, ScheduledTask] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        for _ in range(self._max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self, wait: bool = True) -> None:
        self._running = False
        with self._not_empty:
            self._not_empty.notify_all()
        if wait:
            for w in self._workers:
                w.join(timeout=5)
        self._workers.clear()

    def submit(
        self,
        fn: Callable[..., Any],
        domain: str = "",
        priority: int = 5,
        kwargs: Optional[Dict[str, Any]] = None,
        delay_seconds: float = 0.0,
    ) -> str:
        """Submit a callable for execution.

        Args:
            fn:             The function to call.
            domain:         Logical domain label.
            priority:       0 = highest priority, 9 = lowest.
            kwargs:         Keyword arguments forwarded to *fn*.
            delay_seconds:  Minimum seconds before the task becomes eligible.

        Returns:
            task_id string.
        """
        task = ScheduledTask(
            priority=priority,
            scheduled_at=time.monotonic() + delay_seconds,
            domain=domain,
            fn=fn,
            kwargs=kwargs or {},
        )
        with self._not_empty:
            heapq.heappush(self._heap, task)
            self._task_registry[task.task_id] = task
            self._not_empty.notify()
        return task.task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued task.  Returns True if the task was found and cancelled."""
        with self._lock:
            task = self._task_registry.get(task_id)
            if task and task.status == TaskStatus.QUEUED:
                task.status = TaskStatus.CANCELLED
                return True
        return False

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        task = self._task_registry.get(task_id)
        return task.status if task else None

    def get_result(self, task_id: str) -> Any:
        task = self._task_registry.get(task_id)
        return task.result if task else None

    @property
    def running(self) -> bool:
        return self._running

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._heap if t.status == TaskStatus.QUEUED)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while self._running:
            task = self._pop_next()
            if task is None:
                continue
            task.status = TaskStatus.RUNNING
            task.started_at = time.monotonic()
            try:
                task.result = task.fn(**task.kwargs)
                task.status = TaskStatus.DONE
            except Exception as exc:  # noqa: BLE001
                task.error = str(exc)
                task.status = TaskStatus.FAILED
            finally:
                task.finished_at = time.monotonic()

    def _pop_next(self) -> Optional[ScheduledTask]:
        """Block until a ready task is available, then return it."""
        with self._not_empty:
            while self._running:
                now = time.monotonic()
                # Find the earliest task that is both QUEUED and past its delay
                ready: Optional[ScheduledTask] = None
                for t in self._heap:
                    if t.status == TaskStatus.QUEUED and t.scheduled_at <= now:
                        ready = t
                        break
                if ready is not None:
                    self._heap.remove(ready)
                    heapq.heapify(self._heap)
                    return ready
                # Wait up to 1 s then re-check
                self._not_empty.wait(timeout=1.0)
        return None
