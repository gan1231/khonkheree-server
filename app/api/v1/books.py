"""
Номын CRUD endpoint-ууд.

Нэмэлт (Алхам 1 — Feedback Loop):
  - BookIn.predict_id: identify-book-аас буцаасан predict_id дамжуулбал
    AI дүнийг хэрэглэгчийн мэдээлэлтэй харьцуулж TrainingSample хадгална.

Нэмэлт (Алхам 2 — Vector DB):
  - Ном нэмэх/засах үед cover_url-аас CLIP embedding тооцоолж
    book_embeddings хүснэгтэд хадгална (fire-and-forget background task).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, get_current_user_id
from app.db.models import Book, BookEmbedding, BookStatus, Review, TrainingSample

router = APIRouter(prefix="/books", tags=["books"])


# ─── Pydantic схемүүд ─────────────────────────────────────────────────────────

class BookIn(BaseModel):
    title: str
    author: str
    isbn: Optional[str] = None
    cover_url: Optional[str] = None
    synopsis: Optional[str] = None
    status: BookStatus = BookStatus.owned
    sale_price: Optional[float] = None
    is_public: bool = True
    client_id: uuid.UUID
    # Алхам 1 — Feedback Loop: identify-book-аас буцаасан predict_id
    predict_id: Optional[str] = None


class BookOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    author: str
    isbn: Optional[str]
    cover_url: Optional[str]
    synopsis: Optional[str]
    status: BookStatus
    sale_price: Optional[float]
    is_public: bool
    added_date: datetime
    client_id: uuid.UUID
    review_count: int = 0

    class Config:
        from_attributes = True


# ─── Туслах функцууд ──────────────────────────────────────────────────────────

async def _save_training_sample(
    db: AsyncSession,
    predict_id: str,
    book: Book,
) -> None:
    """
    Алхам 1 — Feedback Loop.
    Redis-ийн кэшлэгдсэн AI дүнтэй хэрэглэгчийн мэдээллийг харьцуулж
    TrainingSample-д хадгална.
    """
    try:
        import redis.asyncio as aioredis
        from app.core.config import settings

        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        raw = await r.get(f"ml:predict:{predict_id}")
        await r.aclose()
    except Exception:
        raw = None

    cached: dict = json.loads(raw) if raw else {}

    ai_title = cached.get("title") or ""
    ai_author = cached.get("author") or ""

    was_corrected = (
        ai_title.strip().lower() != book.title.strip().lower()
        or ai_author.strip().lower() != book.author.strip().lower()
    )

    sample = TrainingSample(
        predict_id=predict_id,
        image_path=cached.get("image_path"),
        ai_title=ai_title or None,
        ai_author=ai_author or None,
        ai_confidence=cached.get("confidence", "low"),
        ai_method=cached.get("method", "vision"),
        correct_title=book.title,
        correct_author=book.author,
        was_corrected=was_corrected,
        needs_review=cached.get("needs_review", False),
        book_id=book.id,
    )
    db.add(sample)
    await db.commit()


async def _upsert_embedding(book_id: uuid.UUID, cover_url: str) -> None:
    """
    Алхам 2 — Vector DB.
    cover_url-аас CLIP embedding тооцоолж book_embeddings хүснэгтэд хадгална.
    Шинэ ном нэмэхэд болон cover_url өөрчлөгдөхөд fire-and-forget дуудагдана.
    """
    from app.ml.embeddings import compute_from_url
    from app.db.session import AsyncSessionLocal
    from sqlalchemy.dialects.postgresql import insert

    vec = await compute_from_url(cover_url)
    if vec is None:
        return

    async with AsyncSessionLocal() as db:
        stmt = (
            insert(BookEmbedding)
            .values(book_id=book_id, embedding=vec, updated_at=datetime.now(timezone.utc))
            .on_conflict_do_update(
                index_elements=["book_id"],
                set_={"embedding": vec, "updated_at": datetime.now(timezone.utc)},
            )
        )
        await db.execute(stmt)
        await db.commit()


# ─── CRUD endpoint-ууд ────────────────────────────────────────────────────────

@router.get("", response_model=list[BookOut])
async def search_books(
    q: Optional[str] = Query(None),
    status: Optional[BookStatus] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Book, func.count(Review.id).label("review_count"))
        .outerjoin(Review, Review.book_id == Book.id)
        .where(Book.is_public.is_(True), Book.deleted_at.is_(None))
        .group_by(Book.id)
        .order_by(Book.added_date.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    if q:
        stmt = stmt.where(or_(Book.title.ilike(f"%{q}%"), Book.author.ilike(f"%{q}%")))
    if status:
        stmt = stmt.where(Book.status == status)

    rows = (await db.execute(stmt)).all()
    result = []
    for book, count in rows:
        out = BookOut.model_validate(book)
        out.review_count = count
        result.append(out)
    return result


@router.get("/matches", response_model=list[BookOut])
async def find_matches_by_query(
    title: str = Query(...),
    my_status: str = Query(...),
    author: Optional[str] = Query(None),
    isbn: Optional[str] = Query(None),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    want_to_read → for_sale, for_exchange тохирох номыг хайна.
    for_sale / for_exchange → want_to_read тохирох номыг хайна.
    """
    status_map = {
        "want_to_read": [BookStatus.for_sale, BookStatus.for_exchange],
        "for_sale": [BookStatus.want_to_read],
        "for_exchange": [BookStatus.want_to_read],
    }
    target_statuses = status_map.get(my_status)
    if not target_statuses:
        return []

    match_conds = []
    if isbn:
        match_conds.append(Book.isbn == isbn)
    if author:
        match_conds.append(
            (func.lower(Book.title) == func.lower(title))
            & (func.lower(Book.author) == func.lower(author))
        )
    else:
        match_conds.append(func.lower(Book.title) == func.lower(title))

    stmt = (
        select(Book, func.count(Review.id).label("review_count"))
        .outerjoin(Review, Review.book_id == Book.id)
        .where(
            Book.user_id != user_id,
            Book.is_public.is_(True),
            Book.deleted_at.is_(None),
            Book.status.in_(target_statuses),
            or_(*match_conds),
        )
        .group_by(Book.id)
        .order_by(Book.added_date.desc())
        .limit(20)
    )
    rows = (await db.execute(stmt)).all()
    result = []
    for book, count in rows:
        out = BookOut.model_validate(book)
        out.review_count = count
        result.append(out)
    return result


@router.get("/mine", response_model=list[BookOut])
async def my_books(
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Book, func.count(Review.id).label("review_count"))
        .outerjoin(Review, Review.book_id == Book.id)
        .where(Book.user_id == user_id, Book.deleted_at.is_(None))
        .group_by(Book.id)
        .order_by(Book.added_date.desc())
    )
    rows = (await db.execute(stmt)).all()
    result = []
    for book, count in rows:
        out = BookOut.model_validate(book)
        out.review_count = count
        result.append(out)
    return result


@router.get("/{book_id}/matches", response_model=list[BookOut])
async def get_book_matches(
    book_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Тодорхой номын ID-аар тохирох номуудыг хайна.
    want_to_read → for_sale, for_exchange
    for_sale / for_exchange → want_to_read
    """
    src = (await db.execute(
        select(Book).where(Book.id == book_id, Book.deleted_at.is_(None))
    )).scalar_one_or_none()
    if not src:
        raise HTTPException(status_code=404, detail="Ном олдсонгүй")

    status_map = {
        BookStatus.want_to_read: [BookStatus.for_sale, BookStatus.for_exchange],
        BookStatus.for_sale: [BookStatus.want_to_read],
        BookStatus.for_exchange: [BookStatus.want_to_read],
    }
    target_statuses = status_map.get(src.status)
    if not target_statuses:
        return []

    match_conds = []
    if src.isbn:
        match_conds.append(Book.isbn == src.isbn)
    match_conds.append(
        (func.lower(Book.title) == func.lower(src.title))
        & (func.lower(Book.author) == func.lower(src.author))
    )

    stmt = (
        select(Book, func.count(Review.id).label("review_count"))
        .outerjoin(Review, Review.book_id == Book.id)
        .where(
            Book.user_id != user_id,
            Book.is_public.is_(True),
            Book.deleted_at.is_(None),
            Book.status.in_(target_statuses),
            or_(*match_conds),
        )
        .group_by(Book.id)
        .order_by(Book.added_date.desc())
        .limit(20)
    )
    rows = (await db.execute(stmt)).all()
    result = []
    for book, count in rows:
        out = BookOut.model_validate(book)
        out.review_count = count
        result.append(out)
    return result


@router.get("/{book_id}", response_model=BookOut)
async def get_book(book_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(Book, func.count(Review.id).label("review_count"))
        .outerjoin(Review, Review.book_id == Book.id)
        .where(Book.id == book_id, Book.deleted_at.is_(None))
        .group_by(Book.id)
    )
    row = (await db.execute(stmt)).first()
    if not row:
        raise HTTPException(status_code=404, detail="Ном олдсонгүй")
    book, count = row
    out = BookOut.model_validate(book)
    out.review_count = count
    return out


@router.post("", response_model=BookOut, status_code=201)
async def add_book(
    body: BookIn,
    background_tasks: BackgroundTasks,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(Book).where(Book.client_id == body.client_id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="client_id давхардсан байна")

    book_data = body.model_dump(exclude={"predict_id"})
    book = Book(**book_data, user_id=user_id)
    db.add(book)
    await db.commit()
    await db.refresh(book)

    # Алхам 1 — Feedback Loop: predict_id байвал TrainingSample хадгална
    if body.predict_id:
        background_tasks.add_task(_save_training_sample, db, body.predict_id, book)

    # Алхам 2 — Vector DB: cover_url байвал CLIP embedding тооцоолно
    if body.cover_url:
        background_tasks.add_task(_upsert_embedding, book.id, body.cover_url)

    return BookOut.model_validate(book)


@router.put("/{book_id}", response_model=BookOut)
async def update_book(
    book_id: uuid.UUID,
    body: BookIn,
    background_tasks: BackgroundTasks,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Book).where(Book.id == book_id, Book.user_id == user_id, Book.deleted_at.is_(None))
    )
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Ном олдсонгүй")

    old_cover = book.cover_url
    for k, v in body.model_dump(exclude={"client_id", "predict_id"}).items():
        setattr(book, k, v)
    book.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(book)

    # Алхам 2 — cover_url өөрчлөгдсөн бол embedding дахин тооцоолно
    if body.cover_url and body.cover_url != old_cover:
        background_tasks.add_task(_upsert_embedding, book.id, body.cover_url)

    return BookOut.model_validate(book)


@router.delete("/{book_id}", status_code=204)
async def delete_book(
    book_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Book).where(Book.id == book_id, Book.user_id == user_id, Book.deleted_at.is_(None))
    )
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Ном олдсонгүй")
    book.deleted_at = datetime.now(timezone.utc)
    await db.commit()
