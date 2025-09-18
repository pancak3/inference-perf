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
from datetime import datetime, timedelta, timezone
from typing import Generator, List, Optional
from polars import read_parquet, DataFrame
from pathlib import Path
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
        except Exception as e:
            raise ValueError(f"Failed to read data from {config.path}: {e}")
        
        if config.start_timestamp is None:
            raise ValueError("start_timestamp must be provided for GeoDistributionDataGenerator")
        if config.first_record_timestamp is None:
            raise ValueError("first_record_timestamp must be provided for GeoDistributionDataGenerator")
        
        self.start_timestamp: datetime = config.start_timestamp
        if self.start_timestamp.tzinfo is None:
            self.start_timestamp = self.start_timestamp.replace(tzinfo=timezone.utc)

        self.first_record_timestamp: datetime = config.first_record_timestamp
        if self.first_record_timestamp.tzinfo is None:
            self.first_record_timestamp = self.first_record_timestamp.replace(tzinfo=timezone.utc)

        self.delay_start_seconds: int = config.delay_start_seconds or 0
        if config.delay_start_seconds is None:
            logger.warning("delay_start_seconds not provided, default to 0")

        # get the time shift in microseconds of seconds
        self.time_shift: timedelta = (self.start_timestamp - self.first_record_timestamp) + timedelta(microseconds=self.delay_start_seconds * 1_000_000)

        logger.info(f"==== GeoDistributionDataGenerator ===="
                    f"\n\tData path: {config.path}"
                    f"\n\tStart timestamp: {self.start_timestamp} (epoch: {self.start_timestamp.timestamp()})"
                    f"\n\tFirst record timestamp: {self.first_record_timestamp} (epoch: {self.first_record_timestamp.timestamp()})"
                    f"\n\tDelay start seconds: {self.delay_start_seconds}"
                    f"\n\tTime shift (microseconds): {self.time_shift.total_seconds() * 1_000_000}")

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.api_config.type != APIType.Chat:
            raise Exception("Unsupported API type")
        
        for row in self.dataset.iter_rows(named=True):
            conversation = row["Conversation"]
            gen_token = row["GeneratedToken"]
            messages = []
            for message in conversation[:-1]:
                messages.append(ChatMessage(role=message["role"], content=message["content"]))
            yield ChatCompletionAPIData(messages=messages, max_completion_tokens=gen_token)

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False
