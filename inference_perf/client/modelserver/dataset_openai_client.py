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

import aiohttp
import json
import time
import logging

logger = logging.getLogger(__name__)


class DatasetOpenAIModelServerClient(vLLMModelServerClient):

    async def process_request(self, data: DatasetChatCompletionAPIData, scheduled_time: float, detailed_result_queue: mp.Queue) -> None:
        assert isinstance(data, DatasetChatCompletionAPIData)
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.api_config.headers:
            headers.update(self.api_config.headers)

        payload = data.to_payload()
        request_data = json.dumps(payload)
        request_id = payload.get("client_side_id", None)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self.max_tcp_connections)) as session:
            start = time.perf_counter()
            try:
                async with session.post(self.uri + data.get_route(), headers=headers, data=request_data) as response:
                    response_info = await data.process_response(
                        response=response, config=self.api_config, tokenizer=self.tokenizer
                    )
                    detailed_result_queue.put((request_id, scheduled_time, start, time.perf_counter(), response_info.output_token_times))
            except Exception as e:
                logger.error("error occured during request processing:", exc_info=True)
        