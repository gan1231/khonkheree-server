from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime, timezone

from app.core.deps import get_db, get_current_user_id
from app.db.models import Conversation, Message, Book

router = APIRouter(tags=["messages"])


class ConversationIn(BaseModel):
    book_id: uuid.UUID
    owner_id: uuid.UUID
    client_id: uuid.UUID


class ConversationOut(BaseModel):
    id: uuid.UUID
    book_id: Optional[uuid.UUID]
    book_title: str
    book_author: str
    initiator_id: uuid.UUID
    owner_id: uuid.UUID
    last_message_at: datetime
    client_id: uuid.UUID

    class Config:
        from_attributes = True


class MessageIn(BaseModel):
    content: str
    client_id: uuid.UUID


class MessageOut(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    sender_id: uuid.UUID
    content: str
    sent_at: datetime
    read_at: Optional[datetime]
    client_id: uuid.UUID

    class Config:
        from_attributes = True


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Conversation)
        .where(or_(Conversation.initiator_id == user_id, Conversation.owner_id == user_id))
        .order_by(Conversation.last_message_at.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [ConversationOut.model_validate(c) for c in rows]


@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def start_conversation(
    body: ConversationIn,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    book = (await db.execute(select(Book).where(Book.id == body.book_id, Book.deleted_at.is_(None)))).scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Ном олдсонгүй")

    existing = (await db.execute(
        select(Conversation).where(
            Conversation.book_id == body.book_id,
            Conversation.initiator_id == user_id,
            Conversation.owner_id == body.owner_id,
        )
    )).scalar_one_or_none()
    if existing:
        return ConversationOut.model_validate(existing)

    conv = Conversation(
        book_id=body.book_id,
        book_title=book.title,
        book_author=book.author,
        initiator_id=user_id,
        owner_id=body.owner_id,
        client_id=body.client_id,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return ConversationOut.model_validate(conv)


@router.get("/conversations/{conv_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conv_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    conv = (await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            or_(Conversation.initiator_id == user_id, Conversation.owner_id == user_id),
        )
    )).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Харилцаа олдсонгүй")

    stmt = select(Message).where(Message.conversation_id == conv_id).order_by(Message.sent_at.asc())
    rows = (await db.execute(stmt)).scalars().all()
    return [MessageOut.model_validate(m) for m in rows]


@router.post("/conversations/{conv_id}/messages", response_model=MessageOut, status_code=201)
async def send_message(
    conv_id: uuid.UUID,
    body: MessageIn,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    conv = (await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            or_(Conversation.initiator_id == user_id, Conversation.owner_id == user_id),
        )
    )).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Харилцаа олдсонгүй")

    msg = Message(conversation_id=conv_id, sender_id=user_id, **body.model_dump())
    conv.last_message_at = datetime.now(timezone.utc)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return MessageOut.model_validate(msg)


@router.put("/conversations/{conv_id}/read", status_code=200)
async def mark_read(
    conv_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    stmt = select(Message).where(
        Message.conversation_id == conv_id,
        Message.sender_id != user_id,
        Message.read_at.is_(None),
    )
    msgs = (await db.execute(stmt)).scalars().all()
    for m in msgs:
        m.read_at = now
    await db.commit()
    return {"updated": len(msgs)}
