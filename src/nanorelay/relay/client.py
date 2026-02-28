import httpx
from nanorelay.schemas.chat import ChatMessage
from typing import AsyncGenerator
import json

class OpenAICompatClient:
    """
    A single HTTP client that forwards requests to any backend implementing the OpenAI /v1/chat/completions specification.
    """

    def __init__(self):
        self._client = httpx.AsyncClient()

    async def chat(
        self,
        request_id: str,
        base_url: str,
        messages: list[ChatMessage],
        model: str,
        timeout: float = 60.0,
    ) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }

        
        response = await self._client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        response_data = response.json()
        response_data["id"] = request_id
        return response_data
    
    async def chat_stream(
            self,
            request_id: str,
            base_url: str,
            messages: list[ChatMessage],
            model: str,
            timeout: float = 60.0,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }

        async with self._client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_part = line[len("data: "):]
                if data_part == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                chunk = json.loads(data_part)
                chunk["id"] = request_id
                if "model" not in chunk or not chunk["model"]:
                    chunk["model"] = model
                yield f"data: {json.dumps(chunk)}\n\n"
                
    async def close(self):
        await self._client.aclose()
