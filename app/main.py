from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1 import auth, books, reviews, messages, ml
from app.ws.handler import ws_connect


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan — серверийн эхлэлт болон зогсолтын үед дуудагдана.
    Алхам 4 (Auto Retrain): APScheduler-ийг энд эхлүүлж, зогсоогдоход унтраана.
    """
    from app.ml.scheduler import start_scheduler, stop_scheduler
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title=settings.APP_NAME, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routers
prefix = settings.API_V1_PREFIX
app.include_router(auth.router, prefix=prefix)
app.include_router(books.router, prefix=prefix)
app.include_router(reviews.router, prefix=prefix)
app.include_router(messages.router, prefix=prefix)
app.include_router(ml.router, prefix=prefix)


# WebSocket — шууд мессеж
@app.websocket("/api/v1/ws/{conv_id}")
async def websocket_endpoint(websocket: WebSocket, conv_id: str, user_id: str):
    await ws_connect(websocket, conv_id, user_id)


@app.get("/health")
async def health():
    return {"status": "ok"}
