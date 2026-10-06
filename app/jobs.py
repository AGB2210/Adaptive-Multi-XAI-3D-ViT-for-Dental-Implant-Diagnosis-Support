"""One worker, one queue.

Everything that touches the model runs here, one job at a time: a single GPU
shared by two concurrent attributions is slower than running them in turn, and
with one model resident there is nothing to race on.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

from src.utils.log import get_logger

log = get_logger("app.jobs")


@dataclass
class Job:
    id: str
    kind: str
    subject: str
    status: str = "queued"           # queued | running | done | failed
    message: str = "Waiting for the worker"
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
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="app-worker", daemon=True)
        self._thread.start()

    def submit(self, kind: str, subject: str, fn) -> Job:
        """Queue `fn(progress)`; `progress(message, fraction)` reports back."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, subject=subject)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, fn))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for(self, subject: str) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values()
                    if j.subject == subject and j.status in ("queued", "running")]

    def _run(self) -> None:
        while True:
            job, fn = self._queue.get()

            def progress(message: str, fraction: float, _job=job) -> None:
                _job.message = message
                _job.progress = max(0.0, min(1.0, float(fraction)))

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
                self._queue.task_done()

    def join(self) -> None:
        """Block until every queued job has finished. For tests."""
        self._queue.join()
