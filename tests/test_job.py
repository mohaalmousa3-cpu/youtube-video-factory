import pytest

from src.core.job import job
from src.database.db import get_connection, init_db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    from src.utils import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.get_settings.cache_clear()
    init_db()
    c = get_connection()
    yield c
    c.close()
    config.get_settings.cache_clear()


def test_job_success_records_output(conn):
    with job(conn, "proj1", "tts", "kokoro") as handle:
        handle.output_path = "data/narration.wav"

    row = conn.execute("SELECT status, output_path FROM jobs WHERE job_id = ?", (handle.job_id,)).fetchone()
    assert row["status"] == "success"
    assert row["output_path"] == "data/narration.wav"


def test_job_failure_records_error_and_reraises(conn):
    with pytest.raises(RuntimeError):
        with job(conn, "proj1", "tts", "kokoro") as handle:
            raise RuntimeError("model file missing")

    row = conn.execute("SELECT status, error FROM jobs WHERE job_id = ?", (handle.job_id,)).fetchone()
    assert row["status"] == "failed"
    assert "model file missing" in row["error"]


def test_job_soft_failure_via_status_override(conn):
    """A job can fail without raising — e.g. TTS produced 0-length audio."""
    with job(conn, "proj1", "tts", "kokoro") as handle:
        handle.status = "failed"
        handle.error = "empty audio output"

    row = conn.execute("SELECT status, error FROM jobs WHERE job_id = ?", (handle.job_id,)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "empty audio output"
