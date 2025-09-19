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
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.apis import InferenceAPIData, InferenceInfo, RequestLifecycleMetric, ErrorResponseInfo
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
import aiohttp
import json
import time
import logging

logger = logging.getLogger(__name__)


class DatasetOpenAIModelServerClient(vLLMModelServerClient):

    async def process_request(self, data: DatasetChatCompletionAPIData, stage_id: int, scheduled_time: float) -> None:
        payload = data.to_payload(
            model_name='',
            max_tokens='',
            ignore_eos='',
            streaming='',
        )
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.api_config.headers:
            headers.update(self.api_config.headers)

        request_data = json.dumps(payload)

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self.max_tcp_connections)) as session:
            start = time.perf_counter()
            try:
                async with session.post(self.uri + data.get_route(), headers=headers, data=request_data) as response:
                    response_info = await data.process_response(
                        response=response, config=self.api_config, tokenizer=self.tokenizer
                    )
                    response_content = await response.text()
                    if response.status == 200:
                        self.metrics_collector.record_metric(
                            RequestLifecycleMetric(
                                stage_id=stage_id,
                                request_data=request_data,
                                response_data=response_content,
                                info=response_info,
                                error=None,
                                start_time=start,
                                end_time=time.perf_counter(),
                                scheduled_time=scheduled_time,
                            )
                        )
                    else:
                        self.metrics_collector.record_metric(
                            RequestLifecycleMetric(
                                stage_id=stage_id,
                                request_data=request_data,
                                response_data=response_content,
                                info=response_info,
                                error=ErrorResponseInfo(error_msg=response_content, error_type="Error response"),
                                start_time=start,
                                end_time=time.perf_counter(),
                                scheduled_time=scheduled_time,
                            )
                        )
            except Exception as e:
                logger.error("error occured during request processing:", exc_info=True)
                self.metrics_collector.record_metric(
                    RequestLifecycleMetric(
                        stage_id=stage_id,
                        request_data=request_data,
                        response_data=response_content if "response_content" in locals() else "",
                        info=response_info if "response_info" in locals() else InferenceInfo(),
                        error=ErrorResponseInfo(
                            error_msg=str(e),
                            error_type=type(e).__name__,
                        ),
                        start_time=start,
                        end_time=time.perf_counter(),
                        scheduled_time=scheduled_time,
                    )
                )

