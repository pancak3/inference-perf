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

from abc import abstractmethod
from inference_perf.apis.dataset_chat import DatasetChatCompletionAPIData
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
import multiprocessing as mp

import asyncio
import aiohttp
import json
import logging
import os
import time
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_LOCAL_QUEUE_STOP = object()
_DEFAULT_LOCAL_QUEUE_SIZE = int(os.getenv("DETAILED_RESULT_LOCAL_QUEUE_SIZE", "50000"))


class DatasetOpenAIModelServerClient(vLLMModelServerClient):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._detailed_result_publisher: Optional[_DetailedResultPublisher] = None
        self._max_retries = 10

    async def process_request(self, data: DatasetChatCompletionAPIData, scheduled_time: float, detailed_result_queue: mp.Queue, retry: int = 0) -> None:
        assert isinstance(data, DatasetChatCompletionAPIData)
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.api_config.headers:
            headers.update(self.api_config.headers)

        payload = data.to_payload()
        max_completion_tokens = payload['max_completion_tokens']
        request_data = json.dumps(payload)
        request_id = payload.get("client_side_id", None)
        # logger.info(f"Sending request id={request_id} at scheduled_time={scheduled_time}")
        publisher = self._ensure_publisher(detailed_result_queue)
        timeout = aiohttp.ClientTimeout(
            total=self.api_config.request_timeout_seconds,
            connect=self.api_config.connect_timeout_seconds,
        )
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.max_tcp_connections),
            timeout=timeout) as session:
            start = time.perf_counter()
            payload_for_logger: Optional[Tuple[Any, ...]] = None
            try:
                async with session.post(self.uri + data.get_route(), headers=headers, data=request_data) as response:
                    response_info = await data.process_response(
                        response=response, config=self.api_config, tokenizer=self.tokenizer
                    )
                    payload_for_logger = (
                        request_id,
                        response.status,
                        scheduled_time,
                        start,
                        time.perf_counter(),
                        response_info.output_token_times,
                        max_completion_tokens,
                    )
            except Exception:
                if retry < self._max_retries:
                    logger.warning(f"Retrying request id={request_id} due to error:", exc_info=True)
                    return await self.process_request(data, scheduled_time, detailed_result_queue, retry=retry+1)
                else:
                    logger.error("error occured during request processing:", exc_info=True)
                    payload_for_logger = (
                        request_id,
                        -1,
                        scheduled_time,
                        start,
                        time.perf_counter(),
                        [],
                        max_completion_tokens,
                    )
            finally:
                if payload_for_logger is not None:
                    publisher.publish(payload_for_logger)
        # logger.info(f"Completed request id={request_id}")

    def _ensure_publisher(self, detailed_result_queue: mp.Queue) -> "_DetailedResultPublisher":
        if (
            self._detailed_result_publisher is None
            or not self._detailed_result_publisher.is_for_queue(detailed_result_queue)
        ):
            self._detailed_result_publisher = _DetailedResultPublisher(detailed_result_queue)
        return self._detailed_result_publisher


class _DetailedResultPublisher:
    def __init__(self, queue: mp.Queue, max_pending: Optional[int] = None) -> None:
        self._target_queue = queue
        pending = _DEFAULT_LOCAL_QUEUE_SIZE if max_pending is None else max_pending
        self._max_pending = max(0, pending)
        self._async_queue: asyncio.Queue[Any]
        if self._max_pending == 0:
            self._async_queue = asyncio.Queue()
        else:
            self._async_queue = asyncio.Queue(maxsize=self._max_pending)
        self._dropped = 0
        self._stopped = False
        loop = asyncio.get_running_loop()
        self._drain_task = loop.create_task(self._drain(loop))

    def is_for_queue(self, queue: mp.Queue) -> bool:
        return self._target_queue is queue

    def publish(self, payload: Tuple[Any, ...]) -> None:
        if self._stopped:
            return
        try:
            self._async_queue.put_nowait(payload)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 1000 == 0:
                logger.warning(
                    "Dropping detailed request metrics due to local buffer overflow (dropped=%s)",
                    self._dropped,
                )

    async def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self._async_queue.put(_LOCAL_QUEUE_STOP)
        await self._drain_task

    async def _drain(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            while True:
                payload = await self._async_queue.get()
                if payload is _LOCAL_QUEUE_STOP:
                    self._async_queue.task_done()
                    break
                try:
                    await loop.run_in_executor(None, self._target_queue.put, payload)
                except Exception:
                    logger.error("Failed to push detailed request metrics to multiprocessing queue", exc_info=True)
                finally:
                    self._async_queue.task_done()
        except asyncio.CancelledError:
            logger.debug("Detailed result publisher drain task cancelled")
            raise