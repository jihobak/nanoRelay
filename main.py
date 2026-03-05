from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from nanorelay.api.v1 import router as v1_router
from nanorelay.core.config import settings
from nanorelay.core.exceptions import NanoRelayException
from nanorelay.relay.dispatcher import Dispatcher

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dispatcher = Dispatcher()
    yield
    await app.state.dispatcher.close()

app = FastAPI(lifespan=lifespan)
app.include_router(v1_router)


@app.exception_handler(NanoRelayException)
async def nanorelay_exception_handler(request: Request, exc: NanoRelayException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": type(exc).__name__
            }
        }
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": type(exc).__name__
            }
        }
    )


if __name__ == "__main__":    
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=True)
