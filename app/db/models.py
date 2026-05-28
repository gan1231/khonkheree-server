import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Boolean, Text, Numeric, Integer, Float,
    ForeignKey, DateTime, Enum as PgEnum, UniqueConstraint, CheckConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
import enum
from app.db.base import Base


def utcnow():
    return datetime.now(timezone.utc)


# ─── Enums ───────────────────────────────────────────────────────────────────

class BookStatus(str, enum.Enum):
    owned = "owned"
    want_to_read = "want_to_read"
    want_to_discuss = "want_to_discuss"
    for_sale = "for_sale"
    for_exchange = "for_exchange"


class SyncAction(str, enum.Enum):
    create = "create"
    update = "update"
    delete = "delete"


class SyncEntity(str, enum.Enum):
    book = "book"
    review = "review"
    message = "message"
    conversation = "conversation"


# ─── Models ──────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    apple_id = Column(String(255), unique=True, nullable=True)
    avatar_url = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    books = relationship("Book", back_populates="user", foreign_keys="Book.user_id")
    reviews = relationship("Review", back_populates="user")


class Book(Base):
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=False)
    author = Column(String(500), nullable=False)
    isbn = Column(String(20), nullable=True)
    cover_url = Column(Text, nullable=True)
    synopsis = Column(Text, nullable=True)
    status = Column(PgEnum(BookStatus), nullable=False, default=BookStatus.owned)
    sale_price = Column(Numeric(10, 2), nullable=True)
    is_public = Column(Boolean, nullable=False, default=True)
    added_date = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
    client_id = Column(UUID(as_uuid=True), unique=True, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="books", foreign_keys=[user_id])
    reviews = relationship("Review", back_populates="book")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    author_name = Column(String(100), nullable=False, default="Нэргүй")
    is_anonymous = Column(Boolean, nullable=False, default=True)
    content = Column(Text, nullable=False)
    likes_count = Column(Integer, nullable=False, default=0)
    client_id = Column(UUID(as_uuid=True), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        CheckConstraint("char_length(content) BETWEEN 1 AND 5000", name="ck_review_content_len"),
    )

    book = relationship("Book", back_populates="reviews")
    user = relationship("User", back_populates="reviews")
    likes = relationship("ReviewLike", back_populates="review")


class ReviewLike(Base):
    __tablename__ = "review_likes"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    review = relationship("Review", back_populates="likes")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id", ondelete="SET NULL"), nullable=True)
    book_title = Column(String(500), nullable=False)
    book_author = Column(String(500), nullable=False)
    initiator_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    last_message_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    client_id = Column(UUID(as_uuid=True), unique=True, nullable=False)

    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    sender_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    sent_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    read_at = Column(DateTime(timezone=True), nullable=True)
    client_id = Column(UUID(as_uuid=True), unique=True, nullable=False)

    __table_args__ = (
        CheckConstraint("char_length(content) BETWEEN 1 AND 2000", name="ck_message_content_len"),
    )

    conversation = relationship("Conversation", back_populates="messages")


class SyncLog(Base):
    __tablename__ = "sync_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    entity_type = Column(PgEnum(SyncEntity), nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    client_id = Column(UUID(as_uuid=True), nullable=False)
    action = Column(PgEnum(SyncAction), nullable=False)
    payload = Column(Text, nullable=True)  # JSON string
    client_timestamp = Column(DateTime(timezone=True), nullable=False)
    synced_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


# ─── Active Learning / Feedback Loop (Алхам 1) ──────────────────────────────

class TrainingSample(Base):
    """
    Хэрэглэгчийн баталгаажуулсан мэдээллийг хадгалах хүснэгт.
    AI-ийн таних дүнтэй харьцуулж, засвар бүрийг сургалтын датасет болгоно.
    """
    __tablename__ = "training_samples"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Redis-ийн кэшийн түлхүүр — identify-book endpoint-аас буцаасан predict_id
    predict_id = Column(String(36), nullable=True, index=True)

    # Зургийн байршил (локал training_images/ хавтас)
    image_path = Column(Text, nullable=True)

    # AI-ийн анхны таних дүн
    ai_title = Column(String(500), nullable=True)
    ai_author = Column(String(500), nullable=True)
    ai_confidence = Column(String(20), nullable=True)   # "low" | "medium" | "high"
    ai_method = Column(String(20), nullable=True)       # "isbn" | "vision" | "db"

    # Хэрэглэгчийн баталгаажуулсан зөв мэдээлэл
    correct_title = Column(String(500), nullable=False)
    correct_author = Column(String(500), nullable=False)

    # AI буруу таньсан эсэх — сургалтанд хамгийн үнэт дохио
    was_corrected = Column(Boolean, nullable=False, default=False)

    # Active Learning: итгэлцэл бага үед True — хэрэглэгчийн баталгаажуулалт шаарддаг (Алхам 3)
    needs_review = Column(Boolean, nullable=False, default=False)

    # Donut сургалтанд ашигласан эсэх
    used_in_training = Column(Boolean, nullable=False, default=False)
    trained_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id", ondelete="SET NULL"), nullable=True)


# ─── Vector DB / CLIP Embeddings (Алхам 2) ──────────────────────────────────

class BookEmbedding(Base):
    """
    Номын хавтасны CLIP зургийн embedding хадгалах хүснэгт.
    sentence-transformers clip-ViT-B-32 загварын 512 хэмжээст вектор.
    Шинэ ном нэмэхэд тэр дороо тооцоологдож хадгалагдана.
    pgvector cosine_distance ашиглан ижил хавтастай номыг хурдан хайна.
    """
    __tablename__ = "book_embeddings"

    book_id = Column(
        UUID(as_uuid=True),
        ForeignKey("books.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(Vector(512), nullable=False)   # CLIP ViT-B/32 — 512 хэмжээ
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
