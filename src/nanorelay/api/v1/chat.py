import time
from typing import Annotated
import uuid
from fastapi import APIRouter, Depends, Request

from nanorelay.core.config import settings
from nanorelay.core.dependencies import get_dispatcher
from nanorelay.core.exceptions import InvalidRequestError
from nanorelay.relay.dispatcher import Dispatcher
from nanorelay.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest, 
    ChatCompletionResponse,  
    MessageRole
)


router = APIRouter(prefix="/chat")


@router.get("/healthz")
async def health_check():
    return {"status": "ok"}


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request, 
    body: ChatCompletionRequest,
    dispatcher: Annotated[Dispatcher, Depends(get_dispatcher)],
):
    request_id = request.headers.get("X-Request-ID") or f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.messages:
        # Extract the last user content as the prompt.
        last_user_msg = body.messages[-1]

        if last_user_msg.role != MessageRole.user:
            raise InvalidRequestError("Last message must have role 'user'")
        
        if not last_user_msg.content.strip():
            raise InvalidRequestError("Prompt cannot be empty")
            
        model_output = await dispatcher.dispatch(body.messages, body.model)
        model_output["id"] = request_id

        return ChatCompletionResponse(**model_output)
