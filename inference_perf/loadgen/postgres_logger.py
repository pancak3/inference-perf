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
import random
import sys
import threading
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Optional, Sequence, cast

import psycopg
from psycopg import sql
from tqdm import tqdm

logger = logging.getLogger(__name__)

_MICROSECONDS_PER_SECOND = 1_000_000
DOWN_SAMPLE_GEO_DATASET = 0

if  "DOWN_SAMPLE_GEO_DATASET" in os.environ:
    try:
        DOWN_SAMPLE_GEO_DATASET = int(os.environ["DOWN_SAMPLE_GEO_DATASET"])
        logger.info(f"Using DOWN_SAMPLE_GEO_DATASET={DOWN_SAMPLE_GEO_DATASET}")
    except ValueError:
        logger.debug("invalid DOWN_SAMPLE_GEO_DATASET value, defaulting to 0")

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

    _QUEUE_TIMEOUT = object()
    _STOP_MARKER = object()

    def __init__(self,num_requests:int,  result_queue: mp.Queue[Any]):
        self._config = self._load_config()
        self._result_queue = result_queue
        self.stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._perf_to_epoch_offset = time.time() - time.perf_counter()
        self._batch_size = self._load_batch_size()
        self._queue_poll_timeout = self._load_queue_poll_timeout()
        self._sleep_bounds = self._load_sleep_bounds()
        self._insert_statement = sql.SQL(
            """
            INSERT INTO {} ("RequestID", "SendAt", "ReceiveFirstTokenAt", "ReceiveLastTokenAt", "MaxCompletionTokens", "GeneratedTokens", "ResponseStatus")
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT ("RequestID") DO UPDATE
            SET "SendAt" = EXCLUDED."SendAt",
                "ReceiveFirstTokenAt" = EXCLUDED."ReceiveFirstTokenAt",
                "ReceiveLastTokenAt" = EXCLUDED."ReceiveLastTokenAt",
                "MaxCompletionTokens" = EXCLUDED."MaxCompletionTokens",
                "GeneratedTokens" = EXCLUDED."GeneratedTokens",
                "ResponseStatus" = EXCLUDED."ResponseStatus"
            """
        ).format(sql.Identifier(self._config.schema, self._config.table))
        self._search_path_statement = sql.SQL("SET search_path TO {};").format(sql.Identifier(self._config.schema))
        self._number_of_requests = num_requests


    def start(self) -> None:
        if self._running:
            return
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
        self.pbar = tqdm(total=self._number_of_requests, desc="Logged requests", position=0, leave=True, disable=False)
        self.stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._run, name="PostgresResultLogger", daemon=True)
        self._running = True
        self._worker_thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        if not self._running:
            return
        self.stop_event.wait()
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

    def _load_batch_size(self) -> int:
        raw_value = os.getenv("INFERENCE_DB_BATCH_SIZE", "500")
        try:
            batch_size = int(raw_value)
        except ValueError as exc:
            raise ValueError("Invalid INFERENCE_DB_BATCH_SIZE value") from exc
        if batch_size <= 0:
            raise ValueError("INFERENCE_DB_BATCH_SIZE must be greater than zero")
        return batch_size

    def _load_queue_poll_timeout(self) -> float:
        raw_value = os.getenv("INFERENCE_DB_QUEUE_POLL_SECONDS", "1.0")
        try:
            timeout = float(raw_value)
        except ValueError as exc:
            raise ValueError("Invalid INFERENCE_DB_QUEUE_POLL_SECONDS value") from exc
        if timeout <= 0:
            raise ValueError("INFERENCE_DB_QUEUE_POLL_SECONDS must be greater than zero")
        return timeout

    def _load_sleep_bounds(self) -> tuple[float, float]:
        min_raw = os.getenv("INFERENCE_DB_SLEEP_MIN_SECONDS", "0.01")
        max_raw = os.getenv("INFERENCE_DB_SLEEP_MAX_SECONDS", "1")
        try:
            minimum = float(min_raw)
            maximum = float(max_raw)
        except ValueError as exc:
            raise ValueError("Invalid INFERENCE_DB_SLEEP_*_SECONDS value") from exc
        if minimum <= 0 or maximum <= 0:
            raise ValueError("Sleep bounds must be positive")
        if maximum < minimum:
            raise ValueError("INFERENCE_DB_SLEEP_MAX_SECONDS must be >= INFERENCE_DB_SLEEP_MIN_SECONDS")
        return (minimum, maximum)

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
        buffered_items: list[Sequence[Any]] = []
        try:
            while True:
                item = self._get_next_item()
                if item is self._STOP_MARKER:
                    if buffered_items:
                        success = self._flush_batch(buffered_items, is_final=True)
                        if success:
                            buffered_items.clear()
                    break
                if item is self._QUEUE_TIMEOUT:
                    if buffered_items:
                        success = self._flush_batch(buffered_items)
                        if success:
                            buffered_items.clear()
                        if not self.stop_event.is_set():
                            self._sleep_with_jitter()
                    continue

                buffered_items.append(cast(Sequence[Any], item))
                if len(buffered_items) >= self._batch_size:
                    success = self._flush_batch(buffered_items)
                    if success:
                        buffered_items.clear()
                    if not self.stop_event.is_set():
                        self._sleep_with_jitter()
        finally:
            self._running = False
            logger.debug("PostgresResultLogger worker stopped")

    def _get_next_item(self) -> object:
        try:
            item = self._result_queue.get(timeout=self._queue_poll_timeout)
        except Empty:
            return self._QUEUE_TIMEOUT
        if item is None:
            return self._STOP_MARKER
        return item

    def _flush_batch(self, items: Sequence[Sequence[Any]], is_final: bool = False) -> bool:
        items_snapshot = list(items)
        if not items_snapshot:
            return True

        payload = [self._transform_item(item) for item in items_snapshot]
        logger.debug("Flushing %d request metrics to Postgres", len(payload))
        try:
            with psycopg.connect(**self._config.connection_kwargs, connect_timeout=self._config.connect_timeout) as conn:
                with conn.cursor() as cur:
                    cur.execute(self._search_path_statement)
                    cur.executemany(self._insert_statement, payload)
                conn.commit()
        except Exception as exc:  # pragma: no cover - runtime error logging
            logger.error("Failed to persist batch of %d metrics: %s", len(payload), exc, exc_info=True)
            return False
        n = len(payload)
        if DOWN_SAMPLE_GEO_DATASET > 0:
            n = n * DOWN_SAMPLE_GEO_DATASET
        self.pbar.update(n)
        if self.pbar.n >= self.pbar.total or self.pbar.n + DOWN_SAMPLE_GEO_DATASET >= self.pbar.total:
            self.pbar.close()
            self.stop_event.set()
        return True

    def _sleep_with_jitter(self) -> None:
        if self.stop_event.is_set():
            return
        lower, upper = self._sleep_bounds
        delay = random.uniform(lower, upper)
        logger.debug("Sleeping %.2f seconds before next DB flush", delay)
        time.sleep(delay)

    def _transform_item(self, item: Sequence[Any]) -> tuple[str, int, int, int, int, int, int]:
        request_id, response_status, _scheduled_time, start, received_at, output_token_times, max_completion_tokens = item
        if request_id is None:
            raise ValueError("RequestID is missing from metrics payload")

        send_at_epoch = self._perf_to_epoch(start)
        if output_token_times:
            first_token_epoch = self._perf_to_epoch(output_token_times[0])
            last_token_epoch = self._perf_to_epoch(output_token_times[-1])
        else:
            first_token_epoch = self._perf_to_epoch(received_at)
            last_token_epoch = self._perf_to_epoch(received_at)

        n_generated_tokens = len(output_token_times)
        return (
            str(request_id),
            to_microseconds(send_at_epoch),
            to_microseconds(first_token_epoch),
            to_microseconds(last_token_epoch),
            max_completion_tokens,
            n_generated_tokens,
            response_status,
        )

    def _perf_to_epoch(self, perf_ts: float) -> float:
        return perf_ts + self._perf_to_epoch_offset
