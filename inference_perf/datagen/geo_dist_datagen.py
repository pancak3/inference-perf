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
from typing import Generator, List, Optional
from polars import read_parquet, DataFrame
from pathlib import Path
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.datagen.base import DataGenerator
from inference_perf.apis import InferenceAPIData, CompletionAPIData, ChatCompletionAPIData, ChatMessage
from inference_perf.utils.custom_tokenizer import CustomTokenizer


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
            pass
        except Exception as e:
            raise ValueError(f"Failed to read data from {config.path}: {e}")


    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.api_config.type != APIType.Chat:
            raise Exception("Unsupported API type")
        
        for row in self.dataset.iter_rows(named=True):
            conversation = row["Conversation"]
            messages = []
            for message in conversation:
                messages.append(ChatMessage(role=message["role"], content=message["content"]))
            yield ChatCompletionAPIData(messages=messages)

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False
