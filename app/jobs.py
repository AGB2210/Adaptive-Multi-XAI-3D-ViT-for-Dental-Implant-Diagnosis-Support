"""Background jobs, run concurrently.

Jobs go to a thread pool and run side by side: two uploaded scans are read,
normalised and located at the same time. Nothing here is sized for a machine.

Two things still run one at a time, and both are about CORRECTNESS, not load:

  per subject  Jobs on the same scan take that scan's lock, because they write
               the same files -- a re-prediction clears the explanations an
               explain job is in the middle of writing.
  per model    Held by `ModelRegistry.lock`, not here: the attribution methods
               register hooks on a model's blocks and switch its attention
               capture on and off, so two of them on one model object would
               read each other's activations.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_all
from dataclasses import dataclass, field

from src.utils.log import get_logger

log = get_logger("app.jobs")


@dataclass
class Job:
    id: str
    kind: str
    subject: str
    status: str = "queued"           # queued | running | done | failed
    message: str = "Queued"
    progress: float = 0.0
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "subject": self.subject,
                "status": self.status, "message": self.message,
                "progress": round(self.progress, 3), "error": self.error,
                "result": self.result}


class JobQueue:
    def __init__(self, workers: int | None = None):
        # None lets the executor pick its own default from the CPU count.
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="app-job")
        self._jobs: dict[str, Job] = {}
        self._futures: list[Future] = []
        self._subject_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._lock = threading.Lock()

    def submit(self, kind: str, subject: str, fn) -> Job:
        """Run `fn(progress)` in the pool; `progress(message, fraction)` reports back."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, subject=subject)
        with self._lock:
            self._jobs[job.id] = job
            subject_lock = self._subject_locks[subject]
            self._futures = [f for f in self._futures if not f.done()]
            self._futures.append(self._pool.submit(self._run, job, fn, subject_lock))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for(self, subject: str) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values()
                    if j.subject == subject and j.status in ("queued", "running")]

    def last_finished_for(self, subject: str, kinds: tuple[str, ...]) -> Job | None:
        """The most recent job of these kinds on `subject` that has ended, done or failed."""
        with self._lock:
            ended = [j for j in self._jobs.values()
                     if j.subject == subject and j.kind in kinds and j.finished is not None]
        return max(ended, key=lambda j: j.finished, default=None)

    def shutdown(self) -> int:
        """Take no more work and drop what has not started.

        Returns how many jobs are still running. A thread cannot be stopped from
        outside, so the caller decides what to do about those; every file a job
        writes goes through a temporary name and a rename, so ending the process
        under one leaves no half-written result behind.
        """
        self._pool.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            return sum(j.status == "running" for j in self._jobs.values())

    def _run(self, job: Job, fn, subject_lock: threading.Lock) -> None:
        def progress(message: str, fraction: float) -> None:
            job.message = message
            job.progress = max(0.0, min(1.0, float(fraction)))

        with subject_lock:
            job.status, job.message = "running", "Starting"
            try:
                job.result = fn(progress)
                job.status, job.progress, job.message = "done", 1.0, "Done"
            except Exception as exc:  # noqa: BLE001 - a failed job is reported, not fatal
                log.error("job %s (%s) failed:\n%s", job.id, job.kind, traceback.format_exc())
                job.status, job.error = "failed", f"{exc}"
                job.message = "Failed"
            finally:
                job.finished = time.time()

    def join(self) -> None:
        """Block until every submitted job has finished. For tests."""
        with self._lock:
            # A job dropped by `shutdown` never runs, and `wait` would sit on it
            # for ever: a cancelled future only counts as done once a worker has
            # picked it up, which is the one thing that will not happen to it.
            pending = [f for f in self._futures if not f.cancelled()]
        wait_all(pending)
