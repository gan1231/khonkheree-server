from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime, timezone

from app.core.deps import get_db, get_current_user_id
from app.db.models import Book, BookStatus, Review

router = APIRouter(prefix="/books", tags=["books"])


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
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(Book).where(Book.client_id == body.client_id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="client_id давхардсан байна")

    book = Book(**body.model_dump(), user_id=user_id)
    db.add(book)
    await db.commit()
    await db.refresh(book)
    return BookOut.model_validate(book)


@router.put("/{book_id}", response_model=BookOut)
async def update_book(
    book_id: uuid.UUID,
    body: BookIn,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Book).where(Book.id == book_id, Book.user_id == user_id, Book.deleted_at.is_(None)))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Ном олдсонгүй")

    for k, v in body.model_dump(exclude={"client_id"}).items():
        setattr(book, k, v)
    book.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(book)
    return BookOut.model_validate(book)


@router.delete("/{book_id}", status_code=204)
async def delete_book(
    book_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Book).where(Book.id == book_id, Book.user_id == user_id, Book.deleted_at.is_(None)))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Ном олдсонгүй")
    book.deleted_at = datetime.now(timezone.utc)
    await db.commit()
