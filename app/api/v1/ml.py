"""
ML endpoint-ууд.

Алхам 1 — Feedback Loop:
  - identify-book: зургийг training_images/-д хадгалж, AI дүнг Redis-д кэшлэнэ
  - feedback: хэрэглэгчийн засварыг training_samples-д бичнэ

Алхам 3 — Active Learning:
  - needs_review: confidence < "high" бол frontend-д баталгаажуулалт хүснэ
  - predict_id: Redis кэшийн түлхүүр (30 минутын TTL)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_db, get_current_user_id

router = APIRouter(prefix="/ml", tags=["ml"])


# ─── Pydantic схемүүд ─────────────────────────────────────────────────────────

class BookIdentifyResult(BaseModel):
    isbn: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    synopsis: Optional[str] = None
    cover_url: Optional[str] = None
    source: str                    # "isbn" | "vision" | "db"
    confidence: float              # 0.0–1.0
    predict_id: str                # Redis кэшийн түлхүүр — books endpoint-д дамжуулна
    needs_review: bool             # True → frontend "баталгаажуулна уу?" дохио харуулна


class FeedbackIn(BaseModel):
    predict_id: str
    correct_title: str
    correct_author: str
    book_id: Optional[str] = None  # Ном хадгалсны дараа холбоно


class FeedbackOut(BaseModel):
    saved: bool
    was_corrected: bool            # AI буруу таньсан байсан эсэх


# ─── Туслах функцууд ──────────────────────────────────────────────────────────

_CONFIDENCE_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3}


def _to_float(confidence: str | None) -> float:
    return _CONFIDENCE_MAP.get((confidence or "").lower(), 0.3)


async def _get_redis():
    """Asyncio Redis клиент (redis.asyncio)."""
    import redis.asyncio as aioredis
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


async def _cache_prediction(predict_id: str, data: dict) -> None:
    """AI дүнг Redis-д PREDICT_CACHE_TTL_SECONDS секундэд хадгална."""
    try:
        r = await _get_redis()
        await r.setex(
            f"ml:predict:{predict_id}",
            settings.PREDICT_CACHE_TTL_SECONDS,
            json.dumps(data, ensure_ascii=False),
        )
        await r.aclose()
    except Exception:
        pass  # Redis байхгүй байсан ч API ажиллаж чадна


async def _load_cached(predict_id: str) -> dict:
    """Redis-ээс кэшлэгдсэн AI дүнг татна. Олдоогүй бол {} буцаана."""
    try:
        r = await _get_redis()
        raw = await r.get(f"ml:predict:{predict_id}")
        await r.aclose()
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


# ─── Endpoint: ном таних ──────────────────────────────────────────────────────

@router.post("/identify-book", response_model=BookIdentifyResult)
async def identify_book(image: UploadFile = File(...)):
    """
    Номын хавтасны зургаас ISBN / гарчиг / зохиогч мэдээллийг буцаана.

    Нэмэлт (Алхам 1+3):
    - Зургийг TRAINING_IMAGES_DIR/{predict_id}.jpg-д хадгална
    - AI дүнг Redis-д 30 минутаар кэшлэнэ
    - predict_id болон needs_review талбараар дамжуулна
    """
    try:
        from app.ml.agent import BookAgent
    except ImportError:
        raise HTTPException(status_code=503, detail="ML модуль ачаалагдаагүй байна")

    suffix = os.path.splitext(image.filename or "img.jpg")[1] or ".jpg"
    content = await image.read()

    # Түр файл үүсгэж agent-д дамжуулна
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    predict_id = str(uuid.uuid4())

    try:
        agent = BookAgent()
        result = agent.identify(tmp_path)

        # Зургийг сургалтын хавтаст хуулах (устгахгүй)
        training_dir = Path(settings.TRAINING_IMAGES_DIR)
        training_dir.mkdir(parents=True, exist_ok=True)
        saved_path = str(training_dir / f"{predict_id}{suffix}")
        shutil.copy2(tmp_path, saved_path)

    finally:
        os.unlink(tmp_path)

    result_dict = result.to_dict()
    confidence_str = result_dict.get("confidence", "low")
    confidence_float = _to_float(confidence_str)

    # Алхам 3 — Active Learning: medium/low итгэлцэл → хэрэглэгчид дохио
    needs_review = confidence_float < 0.85

    authors = result_dict.get("authors") or []
    author_str = authors[0] if authors else None

    # Алхам 1 — Feedback Loop: Redis-д кэшлэх
    await _cache_prediction(predict_id, {
        "title": result_dict.get("title"),
        "author": author_str,
        "confidence": confidence_str,
        "method": result_dict.get("method", "vision"),
        "image_path": saved_path,
        "needs_review": needs_review,
    })

    return BookIdentifyResult(
        isbn=result_dict.get("isbn"),
        title=result_dict.get("title"),
        author=author_str,
        synopsis=result_dict.get("description"),
        cover_url=result_dict.get("cover_url"),
        source=result_dict.get("method", "vision"),
        confidence=confidence_float,
        predict_id=predict_id,
        needs_review=needs_review,
    )


# ─── Endpoint: хэрэглэгчийн засвар (Алхам 1 — Feedback Loop) ────────────────

@router.post("/feedback", response_model=FeedbackOut, status_code=201)
async def submit_feedback(
    body: FeedbackIn,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
):
    """
    Хэрэглэгчийн засварыг сургалтын датасет болгон хадгална.

    Frontend энэ endpoint-ийг ном нэмсний дараа дуудаж болно.
    books endpoint-д predict_id дамжуулбал автоматаар дуудагдана.
    """
    from app.db.models import TrainingSample
    import uuid as _uuid

    cached = await _load_cached(body.predict_id)

    ai_title = cached.get("title") or ""
    ai_author = cached.get("author") or ""

    was_corrected = (
        ai_title.strip().lower() != body.correct_title.strip().lower()
        or ai_author.strip().lower() != body.correct_author.strip().lower()
    )

    book_uuid = None
    if body.book_id:
        try:
            book_uuid = _uuid.UUID(body.book_id)
        except ValueError:
            pass

    sample = TrainingSample(
        predict_id=body.predict_id,
        image_path=cached.get("image_path"),
        ai_title=ai_title or None,
        ai_author=ai_author or None,
        ai_confidence=cached.get("confidence", "low"),
        ai_method=cached.get("method", "vision"),
        correct_title=body.correct_title,
        correct_author=body.correct_author,
        was_corrected=was_corrected,
        needs_review=cached.get("needs_review", False),
        book_id=book_uuid,
    )

    db.add(sample)
    await db.commit()

    return FeedbackOut(saved=True, was_corrected=was_corrected)


# ─── Endpoint: сургалтын статус (хянах хэрэгсэл) ─────────────────────────────

@router.get("/training/status")
async def training_status(db: AsyncSession = Depends(get_db)):
    """Сургалтын датасетийн одоогийн байдлыг буцаана."""
    from sqlalchemy import select, func
    from app.db.models import TrainingSample

    total = (await db.execute(select(func.count(TrainingSample.id)))).scalar_one()
    pending = (await db.execute(
        select(func.count(TrainingSample.id)).where(
            TrainingSample.used_in_training.is_(False)
        )
    )).scalar_one()
    corrected = (await db.execute(
        select(func.count(TrainingSample.id)).where(
            TrainingSample.was_corrected.is_(True)
        )
    )).scalar_one()

    return {
        "total_samples": total,
        "pending_training": pending,
        "corrected_by_user": corrected,
        "retrain_threshold": settings.AUTO_RETRAIN_THRESHOLD,
        "threshold_reached": pending >= settings.AUTO_RETRAIN_THRESHOLD,
    }
