from inference_perf.apis.chat import ChatCompletionAPIData
from typing import Any
from datetime import datetime

class DatasetChatCompletionAPIData(ChatCompletionAPIData):
    request_send_time: datetime
    user_id: str
    conversation_id: str
    turn: int
    max_completion_tokens: int
    model: str
    
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
        }
    