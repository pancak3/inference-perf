import multiprocessing as mp
import time
from typing import Any

import pytest

from inference_perf.loadgen import postgres_logger
from inference_perf.loadgen.postgres_logger import PostgresResultLogger, to_microseconds


class RecordingCursor:
    def __init__(self, connection: "RecordingConnection") -> None:
        self.connection = connection

    def execute(self, statement: Any, params: Any = None) -> None:
        self.connection.executions.append({"statement": statement, "params": params})

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class RecordingConnection:
    def __init__(self) -> None:
        self.executions: list[dict[str, Any]] = []
        self.commits = 0
        self.rollbacks = 0
        self.autocommit = False

    def cursor(self) -> RecordingCursor:
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def __enter__(self) -> "RecordingConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.fixture(autouse=True)
def _restore_time() -> None:
    # Ensure any monkeypatched time functions are restored after each test
    yield
    postgres_logger.time = time


@pytest.fixture
def postgres_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_DB_URL", "postgresql-rw.qifand.com")
    monkeypatch.setenv("INFERENCE_DB_USERNAME", "admin")
    monkeypatch.setenv("INFERENCE_DB_PASSWORD", "secret")
    monkeypatch.setenv("INFERENCE_DB_NAME", "logsdb")
    monkeypatch.setenv("INFERENCE_DB_SCHEMA", "metrics")
    monkeypatch.setenv("INFERENCE_DB_TABLE", "client")
    monkeypatch.setenv("INFERENCE_DB_PORT", "5432")


def test_postgres_logger_persists_records(monkeypatch: pytest.MonkeyPatch, postgres_env: None) -> None:
    connections: list[RecordingConnection] = []

    def fake_connect(*args: Any, **kwargs: Any) -> RecordingConnection:
        conn = RecordingConnection()
        connections.append(conn)
        return conn

    monkeypatch.setattr(postgres_logger.psycopg, "connect", fake_connect)

    monkeypatch.setattr(postgres_logger.time, "time", lambda: 1000.0)
    monkeypatch.setattr(postgres_logger.time, "perf_counter", lambda: 500.0)

    queue: mp.JoinableQueue = mp.JoinableQueue()
    logger = PostgresResultLogger(queue)

    try:
        queue.put(("request-1", 0.0, 10.0, 13.0, [11.0, 12.0]))
        logger.stop(timeout=1.0)
    finally:
        logger.stop(timeout=1.0)
        queue.close()

    assert len(connections) >= 2
    worker_connection = connections[1]
    assert worker_connection.commits == 1

    insert_calls = [call for call in worker_connection.executions if call["params"]]
    assert insert_calls, "expected INSERT execution with parameters"
    params = insert_calls[-1]["params"]
    assert params == ("request-1", 510_000_000, 511_000_000, 512_000_000)


def test_to_microseconds() -> None:
    assert to_microseconds(1.2345) == 1_234_500