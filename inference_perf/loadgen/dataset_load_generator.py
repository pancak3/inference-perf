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
from inference_perf.config import LoadConfig
from inference_perf.loadgen.load_generator import LoadGenerator, StageRuntimeInfo, Status, RequestQueueData
from asyncio import Semaphore, TaskGroup, create_task, gather, run, sleep, set_event_loop_policy
from typing import List, Union
import time
import multiprocessing as mp
import logging
import uvloop

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class DatasetLoadGenerator(LoadGenerator):
    def __init__(self, datagen: GeoDistributionDataGenerator, load_config: LoadConfig) -> None:
        self.datagen = datagen
        self.load_type = load_config.type
        self.stage_runtime_info = dict[int, StageRuntimeInfo]()
        self.num_workers = load_config.num_workers
        self.workers: List[Worker] = []
        self.worker_max_concurrency = load_config.worker_max_concurrency

    async def mp_run(self, client: DatasetOpenAIModelServerClient) -> None:
        request_queue: mp.Queue[RequestQueueData] = mp.JoinableQueue()

        for id in range(self.num_workers):
            self.workers.append(Worker(id, client, request_queue, self.datagen, self.worker_max_concurrency))
            self.workers[-1].start()

        logger.info(f"Multi-workers:{self.num_workers} - run started")

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
        logger.info("Dataset run completed")
        
        for worker in self.workers:
            worker.status_queue.put(Status.WORKER_STOP)

    async def run(self, client: DatasetOpenAIModelServerClient) -> None:
        return await self.mp_run(client)

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
        request_queue: mp.Queue,  # type: ignore[type-arg]
        datagen: GeoDistributionDataGenerator,
        max_concurrency: int,
    ):
        super().__init__()
        self.id = id
        self.client = client
        self.request_queue = request_queue
        self.status_queue: mp.JoinableQueue[Status] = mp.JoinableQueue()
        self.max_concurrency = max_concurrency
        self.datagen = datagen

    def check_status(self) -> Union[Status, None]:
        try:
            return self.status_queue.get_nowait()
        except mp.queues.Empty:
            return None

    async def loop(self) -> None:
        semaphore = Semaphore(self.max_concurrency)
        tasks = []

        while True:
            try:
                await semaphore.acquire()
                item = self.request_queue.get_nowait()

                async def schedule_client(
                    queue: mp.Queue,  # type: ignore[type-arg]
                    request_data: DatasetOpenAIModelServerClient,
                    request_time: float,
                    stage_id: int,
                ) -> None:
                    current_time = time.perf_counter()
                    sleep_time = request_time - current_time
                    if sleep_time > 0:
                        logger.debug(f"Worker {self.id} sleeping for {sleep_time:0.2f} seconds")
                        await sleep(sleep_time)
                    else:
                        logger.debug(f"Worker {self.id} missed scheduled request time by {-1.0 * sleep_time:0.2f}")
                    await self.client.process_request(request_data, stage_id, request_time)
                    queue.task_done()
                    semaphore.release()

                stage_id, request = item
                request_time = request.request_send_time
                task = create_task(schedule_client(self.request_queue, request, request_time, stage_id))
                tasks.append(task)
                await sleep(0)
            except mp.queues.Empty:
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
