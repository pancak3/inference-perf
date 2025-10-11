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
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Generator, List, Optional
from polars import read_parquet, DataFrame
from pathlib import Path
from inference_perf.apis.dataset_chat import DatasetChatCompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.datagen.base import DataGenerator
from inference_perf.apis import InferenceAPIData, CompletionAPIData, ChatCompletionAPIData, ChatMessage
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)

class GeoDistributionDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)
        if not config.path:
            raise ValueError("data_path must be provided for GeoDistributionDataGenerator")
        # if path does not exist
        if Path(config.path).exists() is False:
            raise ValueError(f"Data path {config.path} does not exist")
        try:
            self.dataset: DataFrame = read_parquet(config.path)
            # the datasets are alredy sorted by timestamp
            # self.dataset: DataFrame = self.dataset.sort("Timestamp")
        except Exception as e:
            raise ValueError(f"Failed to read data from {config.path}: {e}")
        
        self.no_wait = config.no_wait
        
        if config.first_record_timestamp is None:
            raise ValueError("first_record_timestamp must be provided for GeoDistributionDataGenerator")
        
        wall_start_ts = config.start_timestamp if config.start_timestamp else datetime.now()
        if wall_start_ts.tzinfo is not None:
            wall_start_ts = wall_start_ts.astimezone(timezone.utc).replace(tzinfo=None)
        wall_start_ts += timedelta(seconds=config.delay_start_seconds)
        
        first_ts_all_geo: datetime = config.first_record_timestamp
        first_ts_all_geo += timedelta(seconds=config.shift_start_seconds)
        self.shift_ts = wall_start_ts - first_ts_all_geo
        self.dataset = self.dataset.filter(self.dataset["Timestamp"] >= first_ts_all_geo)
        self.num_requests = self.dataset.height
    
        if config.duration > 0:
            self.duration = config.duration
            ds_end_ts = first_ts_all_geo + timedelta(seconds=config.duration)
            self.dataset = self.dataset.filter(self.dataset["Timestamp"] <= ds_end_ts)
            self.num_requests = self.dataset.height
        else:
            self.duration = 0
            
        if config.num_requests > 0:
            if config.num_requests > self.num_requests:
                logger.warning(f"Requested number of records {config.num_requests} is greater than available records {self.num_requests}. Using available records.")
            elif config.num_requests < self.num_requests:
                self.dataset = self.dataset.head(config.num_requests)
                logger.info(f"Using only the first {config.num_requests} records from the dataset.")
                self.num_requests = config.num_requests
                
        logger.info(f"Number of records after applying duration and/or number of requests filter: {self.dataset.height}")
        
        
        additional_log = ""
        if len(self.dataset) > 0:
            first_request_ts = self.dataset[0, "Timestamp"] + self.shift_ts
            current_time = datetime.now()
            time_diff = (first_request_ts - current_time).total_seconds()
            additional_log = f"\n\tThe first request will be sent at: {first_request_ts}, {time_diff} seconds from now"
            
        logger.info(f"==== GeoDistributionDataGenerator ===="
            f"\n\tData path: {config.path}"
            f"\n\tExperiment start timestamp: {wall_start_ts}"
            f"\n\tWait for other pods (seconds): {config.delay_start_seconds}"
            f"\n\tFirst record ts among all geo zones: {first_ts_all_geo}"
            f"\n\tShift start seconds: {config.shift_start_seconds}"
            f"\n\tDuration (seconds): {self.duration}"
            f"\n\tNumber of requests: {self.num_requests}"
            f"\n\tNo wait mode: {self.no_wait}"
            f"\n\tTime shift (microseconds): {self.shift_ts}" + additional_log)
        self.adjustment = time.perf_counter() - time.time()


    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        request_send_time = 0
        for row in self.dataset.iter_rows(named=True):
            request_send_time = (row["Timestamp"] + self.shift_ts).timestamp() + self.adjustment
            user_id = str(row["UserID"])
            conversation_id = row["ConversationID"]
            conversation = row["Conversation"]
            turn = row["Turn"]
            max_completion_tokens = row["GeneratedToken"]
            # model = row["Model"]
            model = "Qwen/Qwen3-0.6B" # <- to debug vllm
            record_id = str(row["ID"])
            messages = []
            for message in conversation[:-1]:
                messages.append(ChatMessage(role=message["role"], content=message["content"]))
            yield DatasetChatCompletionAPIData(
                messages=messages, max_completion_tokens=max_completion_tokens, request_send_time=request_send_time, 
                user_id=user_id, conversation_id=conversation_id, turn=turn, model=model, client_side_id=record_id)
        
    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False
