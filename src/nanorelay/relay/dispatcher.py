import json
import time
from typing import AsyncGenerator
from pydantic import BaseModel
from nanorelay.core.config import settings
from nanorelay.core.exceptions import InvalidRequestError
from nanorelay.relay.client import OpenAICompatClient
from nanorelay.schemas.chat import ChatMessage


MODELS_MAP = {
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf": {
        "url": settings.backend_url,
        "timeout": 60.0
    },
}


class BackendConfig(BaseModel):
    url: str
    timeout: float = 60.0


class Dispatcher:
    def __init__(self):
        self._client = OpenAICompatClient()

    async def close(self):
        await self._client.close()

    def _resolve_backend(self, model: str) -> BackendConfig | None:
        if model not in MODELS_MAP:
            raise InvalidRequestError(f"'{model}' is not supported")
        
        config = MODELS_MAP[model]
        if config.get("url") is None:
            return None

        return BackendConfig(**MODELS_MAP[model])
    
    async def _echo_stream(
        self,
        request_id: str,
        model: str,
        content: str,
    ) -> AsyncGenerator[str, None]:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def dispatch(
        self,
        request_id: str,
        messages: list[ChatMessage],
        model: str,
        stream: bool,
    ) -> dict | AsyncGenerator[str, None]:
        backend = self._resolve_backend(model)
        echo_content = f"Echo: {messages[-1].content}"

        if backend is None:
            if stream:
                return self._echo_stream(request_id, model, echo_content)
            else:
                return {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": echo_content},
                        "finish_reason": "stop"
                    }]
                }

        if stream:
            return self._client.chat_stream(
                request_id=request_id,
                base_url=backend.url,
                messages=messages,
                model=model,
                timeout=backend.timeout,
            )

        return await self._client.chat(
            request_id=request_id,
            base_url=backend.url,
            messages=messages,
            model=model,
            timeout=backend.timeout,
        )
