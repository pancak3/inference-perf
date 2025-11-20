import os
import time
from inference_perf.apis.chat import ChatCompletionAPIData
from typing import Any

try:
    TRUNCATE_PROMPT_TOKENS = int(os.getenv("TRUNCATE_PROMPT_TOKENS", "4095"))
except ValueError:
    TRUNCATE_PROMPT_TOKENS = 4095

class DatasetChatCompletionAPIData(ChatCompletionAPIData):
    request_send_time: float
    user_id: str
    conversation_id: str
    turn: int
    model: str
    client_side_id: str
    max_completion_tokens: int = 100
    
    def to_payload(self) -> dict[str, Any]:
        if self.max_completion_tokens == 0:
            self.max_completion_tokens = self.max_completion_tokens
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_completion_tokens": self.max_completion_tokens,
            "ignore_eos": True,
            "stream": True, 
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "turn": self.turn,
            "client_side_id": self.client_side_id,
            "client_sent_at": time.perf_counter(),
            "truncate_prompt_tokens": TRUNCATE_PROMPT_TOKENS - self.max_completion_tokens,
        }
    