# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import psycopg
from psycopg import sql

logger = logging.getLogger(__name__)

_MICROSECONDS_PER_SECOND = 1_000_000


def to_microseconds(timestamp_seconds: float) -> int:
    return int(timestamp_seconds * _MICROSECONDS_PER_SECOND)


@dataclass(frozen=True)
class _DatabaseConfig:
    host: str
    port: int
    username: str
    password: Optional[str]
    database: str
    schema: str
    table: str
    connect_timeout: int = 10

    @property
    def connection_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.username,
        }
        if self.password:
            kwargs["password"] = self.password
        return kwargs


class PostgresResultLogger:
    """Background worker that streams request metrics into PostgreSQL."""

    def __init__(self, result_queue: mp.Queue[Any]):
        self._config = self._load_config()
        self._result_queue = result_queue
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._perf_to_epoch_offset = time.time() - time.perf_counter()
        self._insert_statement = sql.SQL(
            """
            INSERT INTO {} ("RequestID", "SendAt", "ReceiveFirstTokenAt", "ReceiveLastTokenAt")
            VALUES (%s, %s, %s, %s)
            ON CONFLICT ("RequestID") DO UPDATE
            SET "SendAt" = EXCLUDED."SendAt",
                "ReceiveFirstTokenAt" = EXCLUDED."ReceiveFirstTokenAt",
                "ReceiveLastTokenAt" = EXCLUDED."ReceiveLastTokenAt"
            """
        ).format(sql.Identifier(self._config.schema, self._config.table))
        self._search_path_statement = sql.SQL("SET search_path TO {};").format(sql.Identifier(self._config.schema))

        self._test_connection()
        logger.info(
            "Connected to Postgres host=%s port=%s db=%s schema=%s table=%s",
            self._config.host,
            self._config.port,
            self._config.database,
            self._config.schema,
            self._config.table,
        )
        atexit.register(self.stop)
        self.start()

    def start(self) -> None:
        if self._running:
            return
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._run, name="PostgresResultLogger", daemon=True)
        self._running = True
        self._worker_thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._worker_thread:
            if self._worker_thread.is_alive():
                try:
                    self._result_queue.put_nowait(None)
                except Exception:
                    pass
            self._worker_thread.join(timeout=timeout)
        self._worker_thread = None
        self._running = False

    def _load_config(self) -> _DatabaseConfig:
        try:
            port = int(os.getenv("INFERENCE_DB_PORT", "5432"))
        except ValueError as exc:
            raise ValueError("Invalid INFERENCE_DB_PORT value") from exc

        username = os.getenv("INFERENCE_DB_USERNAME")
        if not username:
            raise ValueError("INFERENCE_DB_USERNAME environment variable is required")

        database = os.getenv("INFERENCE_DB_NAME")
        if not database:
            raise ValueError("INFERENCE_DB_NAME environment variable is required")

        schema = os.getenv("INFERENCE_DB_SCHEMA")
        if not schema:
            raise ValueError("INFERENCE_DB_SCHEMA environment variable is required")

        table = os.getenv("INFERENCE_DB_TABLE")
        if not table:
            raise ValueError("INFERENCE_DB_TABLE environment variable is required")

        host = os.getenv("INFERENCE_DB_URL")
        if not host:
            raise ValueError("INFERENCE_DB_URL environment variable is required")

        password = os.getenv("INFERENCE_DB_PASSWORD")
        password_value = password if password else None

        return _DatabaseConfig(
            host=host,
            port=port,
            username=username,
            password=password_value,
            database=database,
            schema=schema,
            table=table,
        )

    def _test_connection(self) -> None:
        try:
            with psycopg.connect(**self._config.connection_kwargs, connect_timeout=self._config.connect_timeout) as conn:
                with conn.cursor() as cur:
                    cur.execute(self._search_path_statement)
        except Exception as exc:  # pragma: no cover - fatal path
            logger.error("Failed to connect to Postgres: %s", exc, exc_info=True)
            sys.exit(1)

    def _run(self) -> None:
        logger.debug("PostgresResultLogger worker started")
        try:
            with psycopg.connect(**self._config.connection_kwargs, connect_timeout=self._config.connect_timeout) as conn:
                conn.autocommit = False
                with conn.cursor() as cur:
                    cur.execute(self._search_path_statement)
                    while True:
                        item = self._result_queue.get()
                        if item is None:
                            break
                        try:
                            request_id, send_at, first_token_at, last_token_at = self._transform_item(item)
                            cur.execute(
                                self._insert_statement,
                                (request_id, send_at, first_token_at, last_token_at),
                            )
                            conn.commit()
                        except Exception as exc:  # pragma: no cover - runtime error logging
                            conn.rollback()
                            logger.error("Failed to persist metrics for request %s: %s", item[0], exc, exc_info=True)
        finally:
            self._running = False
            logger.debug("PostgresResultLogger worker stopped")

    def _transform_item(self, item: Sequence[Any]) -> tuple[str, int, int, int]:
        request_id, _scheduled_time, start, received_at, output_token_times = item
        if request_id is None:
            raise ValueError("RequestID is missing from metrics payload")

        send_at_epoch = self._perf_to_epoch(start)
        if output_token_times:
            first_token_epoch = self._perf_to_epoch(output_token_times[0])
            last_token_epoch = self._perf_to_epoch(output_token_times[-1])
        else:
            first_token_epoch = self._perf_to_epoch(received_at)
            last_token_epoch = self._perf_to_epoch(received_at)

        return (
            str(request_id),
            to_microseconds(send_at_epoch),
            to_microseconds(first_token_epoch),
            to_microseconds(last_token_epoch),
        )

    def _perf_to_epoch(self, perf_ts: float) -> float:
        return perf_ts + self._perf_to_epoch_offset
