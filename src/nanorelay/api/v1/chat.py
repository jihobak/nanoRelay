import time
import uuid
from fastapi import APIRouter, Request

from nanorelay.core.config import settings
from nanorelay.core.exceptions import InvalidRequestError
from nanorelay.schemas.chat import (
    ChatCompletionMessage, 
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionChoice, 
    MessageRole
)


router = APIRouter(prefix="/chat")


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    request_id = request.headers.get("X-Request-ID") or f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.messages:
        # Extract the last user content as the prompt.
        last_user_msg = body.messages[-1]
        prompt = last_user_msg.content
        role = last_user_msg.role

        if role != MessageRole.user:
            raise InvalidRequestError("Last message must have role 'user'")
                
        if settings.backend_url is None:
            return ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=body.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionMessage(
                            role=MessageRole.assistant,
                            content=f"Echo: {prompt}"
                        ),
                        finish_reason="stop"
                    ),
                ]
            )
    