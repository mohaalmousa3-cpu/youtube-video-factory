"""Job bookkeeping: record a unit of work (one provider call) against a
project so a failed step can be retried alone instead of redoing everything
before it — the idempotency rule from the plan."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional


class JobHandle:
    __slots__ = ("job_id", "status", "output_path", "error")

    def __init__(self, job_id: int):
        self.job_id = job_id
        self.status: Optional[str] = None
        self.output_path: Optional[str] = None
        self.error: Optional[str] = None


@contextmanager
def job(conn: sqlite3.Connection, project_id: str, job_type: str, provider: str) -> Iterator[JobHandle]:
    """Wrap one provider call: records queued -> running -> success/failed.
    On success (no exception, .status left as None) marks 'success' and
    stores .output_path; set handle.status = 'failed' with .error to record
    a soft failure without raising."""
    cursor = conn.execute(
        "INSERT INTO jobs (project_id, job_type, provider, status) VALUES (?, ?, ?, 'queued')",
        (project_id, job_type, provider),
    )
    conn.commit()
    handle = JobHandle(cursor.lastrowid)
    conn.execute("UPDATE jobs SET status = 'running', started_at = datetime('now') WHERE job_id = ?", (handle.job_id,))
    conn.commit()
    try:
        yield handle
    except Exception as exc:
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, finished_at = datetime('now') WHERE job_id = ?",
            (str(exc), handle.job_id),
        )
        conn.commit()
        raise
    else:
        final_status = handle.status or "success"
        conn.execute(
            "UPDATE jobs SET status = ?, output_path = ?, error = ?, finished_at = datetime('now') WHERE job_id = ?",
            (final_status, handle.output_path, handle.error, handle.job_id),
        )
        conn.commit()
