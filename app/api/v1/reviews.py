from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime, timezone

from app.core.deps import get_db, get_current_user_id
from app.db.models import Review, ReviewLike, Book

router = APIRouter(tags=["reviews"])


class ReviewIn(BaseModel):
    content: str
    author_name: str = "Нэргүй"
    is_anonymous: bool = True
    client_id: uuid.UUID


class ReviewOut(BaseModel):
    id: uuid.UUID
    book_id: uuid.UUID
    user_id: Optional[uuid.UUID]
    author_name: str
    is_anonymous: bool
    content: str
    likes_count: int
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/books/{book_id}/reviews", response_model=list[ReviewOut])
async def list_reviews(book_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(Review).where(Review.book_id == book_id).order_by(Review.created_at.desc())
    rows = (await db.execute(stmt)).scalars().all()
    return [ReviewOut.model_validate(r) for r in rows]


@router.post("/books/{book_id}/reviews", response_model=ReviewOut, status_code=201)
async def add_review(
    book_id: uuid.UUID,
    body: ReviewIn,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[uuid.UUID] = None,  # нэвтрэлтгүйгээр боломжтой
):
    book = (await db.execute(select(Book).where(Book.id == book_id, Book.deleted_at.is_(None)))).scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Ном олдсонгүй")

    existing = (await db.execute(select(Review).where(Review.client_id == body.client_id))).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="client_id давхардсан байна")

    review = Review(
        book_id=book_id,
        user_id=user_id,
        **body.model_dump(),
    )
    db.add(review)
    await db.commit()
    await db.refresh(review)
    return ReviewOut.model_validate(review)


@router.post("/reviews/{review_id}/like", status_code=200)
async def toggle_like(
    review_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    review = (await db.execute(select(Review).where(Review.id == review_id))).scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Сэтгэгдэл олдсонгүй")

    existing = (
        await db.execute(select(ReviewLike).where(ReviewLike.user_id == user_id, ReviewLike.review_id == review_id))
    ).scalar_one_or_none()

    if existing:
        await db.delete(existing)
        review.likes_count = max(0, review.likes_count - 1)
        liked = False
    else:
        db.add(ReviewLike(user_id=user_id, review_id=review_id))
        review.likes_count += 1
        liked = True

    await db.commit()
    return {"liked": liked, "likes_count": review.likes_count}
