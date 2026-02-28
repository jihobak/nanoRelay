import time
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

    def _resolve_backend(self, model: str) -> BackendConfig:
        if model not in MODELS_MAP:
            raise InvalidRequestError(f"'{model}' is not supported")
        
        config = MODELS_MAP[model]
        if config.get("url") is None:
            return None

        return BackendConfig(**MODELS_MAP[model])

    async def dispatch(self, messages: list[ChatMessage], model: str) -> dict:
        backend = self._resolve_backend(model)

        if backend is None:
            # this happend when 'NANORELAY_BACKEND_URL' is not set
            return {
                "id": "",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Echo: {messages[-1].content}"},
                    "finish_reason": "stop"
                }]
            }

        return await self._client.chat(
            base_url=backend.url,
            messages=messages,
            model=model,
            timeout=backend.timeout,
        )
