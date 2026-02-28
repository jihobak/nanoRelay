import httpx
from nanorelay.schemas.chat import ChatMessage


class OpenAICompatClient:
    """
    A single HTTP client that forwards requests to any backend implementing the OpenAI /v1/chat/completions specification.
    """

    async def chat(
        self,
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

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            return response.json()
