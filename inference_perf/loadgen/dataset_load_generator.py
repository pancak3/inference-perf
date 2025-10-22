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

import logging
import multiprocessing as mp
import time
import uvloop
from datetime import datetime,timezone
from asyncio import Semaphore, create_task, gather, run, sleep, set_event_loop_policy
from queue import Empty
from typing import List, Union

from inference_perf.apis.dataset_chat import DatasetChatCompletionAPIData
from inference_perf.client.modelserver.dataset_openai_client import DatasetOpenAIModelServerClient
from inference_perf.datagen import GeoDistributionDataGenerator
from inference_perf.config import LoadConfig, StorageConfigBase
from inference_perf.loadgen.load_generator import LoadGenerator, StageRuntimeInfo, Status
from inference_perf.loadgen.postgres_logger import PostgresResultLogger

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class DatasetLoadGenerator(LoadGenerator):
    def __init__(self, datagen: GeoDistributionDataGenerator, load_config: LoadConfig, local_storage: StorageConfigBase) -> None:
        self.datagen = datagen
        self.load_type = load_config.type
        self.stage_runtime_info = dict[int, StageRuntimeInfo]()
        self.num_workers = load_config.num_workers
        self.workers: List[Worker] = []
        self.worker_max_concurrency = load_config.worker_max_concurrency
        self.local_storage = local_storage
        self.detailed_result_queue: mp.JoinableQueue = mp.JoinableQueue()
        self.result_logger = PostgresResultLogger(self.datagen.dataset.height, self.detailed_result_queue)

    async def mp_run(self, client: DatasetOpenAIModelServerClient) -> None:
        request_queue: mp.JoinableQueue = mp.JoinableQueue()

        for id in range(self.num_workers):
            self.workers.append(Worker(id, client, request_queue, self.datagen, self.worker_max_concurrency, self.detailed_result_queue))
            self.workers[-1].start()

        # Allow generation a second to begin populating the queue so the workers
        # don't miss the initial scheuled request times
        start_time_epoch = datetime.now(timezone.utc).timestamp() + 1

        data_generator = self.datagen.get_data()
        n = 0
        for item in data_generator:
            request_queue.put((0, item))
            n += 1
        logger.debug(f"Loaded {n} requests into the queue")

        # Join on request queue to ensure that all workers have completed
        # their requests for the stage
        while request_queue.qsize() > 0:
            logger.debug(f"Loadgen awaiting empty request queue, current size: {request_queue.qsize()}")
            await sleep(1)

        logger.debug("Loadgen sending STAGE_END to workers")
        for worker in self.workers:
            worker.status_queue.put(Status.STAGE_END)

        for worker in self.workers:
            while worker.status_queue.qsize() > 0:
                logger.debug(f"Loadgen waiting for worker {worker.id} to process STAGE_END")
                await sleep(1)
            worker.status_queue.join()

        logger.debug("Loadgen joining request queue")
        request_queue.join()
        self.stage_runtime_info[0] = StageRuntimeInfo(
            stage_id=0, rate=-1, start_time=start_time_epoch, end_time=datetime.now(timezone.utc).timestamp()
        )
        
        for worker in self.workers:
            worker.status_queue.put(Status.WORKER_STOP)

    async def run(self, client: DatasetOpenAIModelServerClient) -> None:
        if self.datagen.dataset.height > 0:
            self.result_logger.start()
        try:
            await self.mp_run(client)
        finally:
            if self.datagen.dataset.height > 0:
                self.result_logger.stop()

    async def stop(self) -> None:
        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.0)
                
class Worker(mp.Process):
    def __init__(
        self,
        id: int,
        client: DatasetOpenAIModelServerClient,
    request_queue: mp.JoinableQueue,  # type: ignore[type-arg]
        datagen: GeoDistributionDataGenerator,
        max_concurrency: int,
    detailed_result_queue: mp.JoinableQueue,
    ):
        super().__init__()
        self.id = id
        assert isinstance(client, DatasetOpenAIModelServerClient)
        self.client = client
        self.request_queue = request_queue
        self.status_queue = mp.JoinableQueue()
        self.max_concurrency = max_concurrency
        self.datagen = datagen
        self.detailed_result_queue = detailed_result_queue

    def check_status(self) -> Union[Status, None]:
        try:
            return self.status_queue.get_nowait()
        except Empty:
            return None

    async def loop(self) -> None:
        semaphore = Semaphore(self.max_concurrency)
        tasks = []

        while True:
            try:
                await semaphore.acquire()
                item = self.request_queue.get_nowait()

                async def schedule_client(
                    queue: mp.JoinableQueue,  # type: ignore[type-arg]
                    request_data: DatasetChatCompletionAPIData,
                    request_time: float,
                    detailed_result_queue: mp.JoinableQueue
                ) -> None:
                    current_time = time.perf_counter()
                    sleep_time = request_time - current_time if (not self.datagen.no_wait) else 0
                    if sleep_time > 0:
                        logger.debug(f"Worker {self.id} sleeping for {sleep_time:0.2f} seconds")
                        await sleep(sleep_time)
                    else:
                        logger.debug(f"Worker {self.id} missed scheduled request time by {-1.0 * sleep_time:0.2f}")
                    await self.client.process_request(request_data, request_time, detailed_result_queue)
                    queue.task_done()
                    semaphore.release()

                _, request = item
                request_time = request.request_send_time
                task = create_task(schedule_client(self.request_queue, request, request_time, self.detailed_result_queue))
                tasks.append(task)
                await sleep(0)
            except Empty:
                semaphore.release()
                status = self.check_status()
                if status is None:
                    await sleep(0)
                if status is not None:
                    logger.debug(f"[Worker {self.id}] received {status}, awaiting {len(tasks)} tasks")
                    await gather(*tasks)
                    tasks = []
                    self.status_queue.task_done()
                if status == Status.STAGE_END:
                    continue
                if status == Status.WORKER_STOP:
                    break

    def run(self) -> None:
        set_event_loop_policy(uvloop.EventLoopPolicy())
        run(self.loop())