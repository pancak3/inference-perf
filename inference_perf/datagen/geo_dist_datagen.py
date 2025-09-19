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
        
        if config.first_record_timestamp is None:
            raise ValueError("first_record_timestamp must be provided for GeoDistributionDataGenerator")
        
        start_ts = config.start_timestamp if config.start_timestamp else datetime.now(timezone.utc)
        if start_ts.tzinfo is None:
           start_ts = start_ts.replace(tzinfo=timezone.utc)
           
        if config.delay_start_seconds:
            logger.info(f"Adjusting start time by {config.delay_start_seconds} seconds")
            start_ts = start_ts + timedelta(seconds=config.delay_start_seconds)
        first_record_ts: datetime = config.first_record_timestamp
        if first_record_ts.tzinfo is None:
            first_record_ts = first_record_ts.replace(tzinfo=timezone.utc)

        shift_ts = start_ts - first_record_ts
        # get the time shift in microseconds of seconds

        logger.info(f"==== GeoDistributionDataGenerator ===="
                    f"\n\tData path: {config.path}"
                    f"\n\tStart timestamp: {start_ts}"
                    f"\n\tFirst record timestamp: {first_record_ts}"
                    f"\n\tDelay start seconds: {config.delay_start_seconds}"
                    f"\n\tTime shift (microseconds): {shift_ts}")
        self.start_ts = start_ts
        self.first_record_ts = first_record_ts
        self.shift_ts = shift_ts
        self.adjustment = time.perf_counter() - datetime.now(timezone.utc).timestamp()
        self.num_requests = config.num_requests if config.num_requests else self.dataset.height
        if self.num_requests > self.dataset.height:
            logger.warning(f"Requested number of records {self.num_requests} exceeds dataset size {self.dataset.height}, using {self.dataset.height}")
            self.num_requests = self.dataset.height
        self.duration = config.duration if config.duration else 0
        self.end_ts = self.start_ts.timestamp() + self.duration + self.adjustment if self.duration > 0 else 0
        self.no_wait = config.no_wait

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.api_config.type != APIType.Chat:
            raise Exception("Unsupported API type")
        count = 0
        request_send_time = 0
        for row in self.dataset.iter_rows(named=True):
            request_send_time = (row["Timestamp"] + self.shift_ts).replace(tzinfo=timezone.utc).timestamp() + self.adjustment
            if self.end_ts > 0 and request_send_time >= self.end_ts:
                if count < self.num_requests:
                    self.num_requests = count
                    logger.warning(f"Only {count} requests generated before reaching the duration limit of {self.duration} seconds")
                break
            user_id = str(row["UserID"])
            conversation_id = row["ConversationID"]
            conversation = row["Conversation"]
            turn = row["Turn"]
            max_completion_tokens = row["GeneratedToken"]
            model = row["Model"]
            record_id = row["ID"]
            messages = []
            for message in conversation[:-1]:
                messages.append(ChatMessage(role=message["role"], content=message["content"]))
            yield DatasetChatCompletionAPIData(
                messages=messages, max_completion_tokens=max_completion_tokens, request_send_time=request_send_time, 
                user_id=user_id, conversation_id=conversation_id, turn=turn, model=model, id=record_id)
            count += 1
            if count >= self.num_requests:
                break

        
    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False
