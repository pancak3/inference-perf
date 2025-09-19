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
from datetime import datetime,timezone
from inference_perf.client.modelserver.dataset_openai_client import DatasetOpenAIModelServerClient
from inference_perf.datagen import GeoDistributionDataGenerator
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.config import LoadConfig
from inference_perf.loadgen.load_generator import LoadGenerator, StageRuntimeInfo, Worker, Status, RequestQueueData
from asyncio import  TaskGroup, create_task, gather, run, sleep, set_event_loop_policy
from typing import List
import time
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)

class DatasetLoadGenerator(LoadGenerator):
    def __init__(self, datagen: GeoDistributionDataGenerator, load_config: LoadConfig) -> None:
        self.datagen = datagen
        self.load_type = load_config.type
        self.stage_runtime_info = dict[int, StageRuntimeInfo]()
        self.num_workers = load_config.num_workers
        self.workers: List[Worker] = []
        self.worker_max_concurrency = load_config.worker_max_concurrency
        # dataset 
        self.num_requests = load_config.num_requests
        self.duration = load_config.duration

    async def mp_run(self, client: ModelServerClient) -> None:
        request_queue: mp.Queue[RequestQueueData] = mp.JoinableQueue()

        for id in range(self.num_workers):
            self.workers.append(Worker(id, client, request_queue, self.datagen, self.worker_max_concurrency))
            self.workers[-1].start()

        for stage_id, stage in enumerate(self.stages):
            logger.info("Stage %d - run started", stage_id)
            timer = self.get_timer(stage.rate, stage.duration)

            # Allow generation a second to begin populating the queue so the workers
            # don't miss the initial scheuled request times
            start_time_epoch = time.time()
            start_time = time.perf_counter() + 1
            num_requests = int(stage.rate * stage.duration)
            start_time_epoch = time.time()

            time_generator = timer.start_timer(start_time)
            if hasattr(self.datagen, "get_request"):
                # Datagen supports deferring to workers, enqueue request number
                for request_number in range(num_requests):
                    request_time = next(time_generator)
                    request_queue.put((stage_id, request_number, request_time))
            else:
                # Datagen requires queueing request_data
                data_generator = self.datagen.get_data()
                for _ in range(num_requests):
                    request_queue.put((stage_id, next(data_generator), next(time_generator)))

            await sleep(start_time + stage.duration - time.perf_counter())

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
            self.stage_runtime_info[stage_id] = StageRuntimeInfo(
                stage_id=stage_id, rate=stage.rate, start_time=start_time_epoch, end_time=time.time()
            )
            logger.info("Stage %d - run completed", stage_id)
            if self.stageInterval and stage_id < len(self.stages) - 1:
                await sleep(self.stageInterval)

        for worker in self.workers:
            worker.status_queue.put(Status.WORKER_STOP)

    async def run(self, client: DatasetOpenAIModelServerClient) -> None:
        # if self.num_workers > 0:
        #     return await self.mp_run(client)

        start_time_epoch = self.datagen.start_ts.timestamp()
        if self.duration:
            adjustment = time.perf_counter() - datetime.now(timezone.utc).timestamp()
            end_time = start_time_epoch + adjustment + self.duration
        logger.info("Dataset - run started")
        async with TaskGroup() as tg:
            n_sent = 0
            for chat in self.datagen.get_data():
                now = time.perf_counter()
                if now > chat.request_send_time:
                    tg.create_task(client.process_request(chat, 0, now))
                else:
                    time_to_wait = chat.request_send_time - now
                    if now + time_to_wait > end_time:
                        break
                    await sleep(time_to_wait)
                    tg.create_task(client.process_request(chat, 0, chat.request_send_time))
                if self.duration and now > end_time:
                    break
                n_sent += 1
                if self.num_requests and n_sent >= self.num_requests:
                    break
        self.stage_runtime_info[0] = StageRuntimeInfo(
            stage_id=0, rate=-1, start_time=start_time_epoch, end_time=datetime.now(timezone.utc).timestamp()
        )
        logger.info("All requests sent, waiting for completion")

    async def stop(self) -> None:
        for worker in self.workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.0)