"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-17
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=True),
        sa.Column("apple_id", sa.String(255), nullable=True, unique=True),
        sa.Column("avatar_url", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_users_email", "users", ["email"])
    op.create_index("idx_users_apple_id", "users", ["apple_id"], postgresql_where=sa.text("apple_id IS NOT NULL"))

    op.create_table(
        "books",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("author", sa.String(500), nullable=False),
        sa.Column("isbn", sa.String(20), nullable=True),
        sa.Column("cover_url", sa.Text, nullable=True),
        sa.Column("synopsis", sa.Text, nullable=True),
        sa.Column("status", sa.Enum("owned","want_to_read","want_to_discuss","for_sale","for_exchange", name="book_status"), nullable=False, server_default="owned"),
        sa.Column("sale_price", sa.Numeric(10, 2), nullable=True),
        sa.Column("is_public", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("added_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("client_id", UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_books_user_id", "books", ["user_id"])
    op.create_index("idx_books_status", "books", ["status"], postgresql_where=sa.text("is_public = true"))
    op.execute("CREATE INDEX idx_books_search ON books USING GIN(to_tsvector('simple', title || ' ' || author))")

    op.create_table(
        "reviews",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("book_id", UUID(as_uuid=True), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("author_name", sa.String(100), nullable=False, server_default="'Нэргүй'"),
        sa.Column("is_anonymous", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("likes_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("client_id", UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("char_length(content) BETWEEN 1 AND 5000", name="ck_review_content_len"),
    )
    op.create_index("idx_reviews_book_id", "reviews", ["book_id"])

    op.create_table(
        "review_likes",
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("review_id", UUID(as_uuid=True), sa.ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # Auto-update likes_count trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION update_likes_count() RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                UPDATE reviews SET likes_count = likes_count + 1 WHERE id = NEW.review_id;
            ELSIF TG_OP = 'DELETE' THEN
                UPDATE reviews SET likes_count = GREATEST(0, likes_count - 1) WHERE id = OLD.review_id;
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        CREATE TRIGGER trg_review_likes
        AFTER INSERT OR DELETE ON review_likes
        FOR EACH ROW EXECUTE FUNCTION update_likes_count();
    """)

    op.create_table(
        "conversations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("book_id", UUID(as_uuid=True), sa.ForeignKey("books.id", ondelete="SET NULL"), nullable=True),
        sa.Column("book_title", sa.String(500), nullable=False),
        sa.Column("book_author", sa.String(500), nullable=False),
        sa.Column("initiator_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("owner_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("client_id", UUID(as_uuid=True), nullable=False, unique=True),
    )
    op.create_index("idx_conv_initiator", "conversations", ["initiator_id"])
    op.create_index("idx_conv_owner", "conversations", ["owner_id"])

    op.create_table(
        "messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sender_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("client_id", UUID(as_uuid=True), nullable=False, unique=True),
        sa.CheckConstraint("char_length(content) BETWEEN 1 AND 2000", name="ck_message_content_len"),
    )
    op.create_index("idx_messages_conv", "messages", ["conversation_id"])

    op.execute("""
        CREATE OR REPLACE FUNCTION update_conversation_last_message() RETURNS TRIGGER AS $$
        BEGIN
            UPDATE conversations SET last_message_at = NEW.sent_at WHERE id = NEW.conversation_id;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        CREATE TRIGGER trg_update_last_message
        AFTER INSERT ON messages
        FOR EACH ROW EXECUTE FUNCTION update_conversation_last_message();
    """)

    op.create_table(
        "sync_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entity_type", sa.Enum("book","review","message","conversation", name="sync_entity"), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), nullable=False),
        sa.Column("client_id", UUID(as_uuid=True), nullable=False),
        sa.Column("action", sa.Enum("create","update","delete", name="sync_action"), nullable=False),
        sa.Column("payload", sa.Text, nullable=True),
        sa.Column("client_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("sync_log")
    op.execute("DROP TRIGGER IF EXISTS trg_update_last_message ON messages")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.execute("DROP TRIGGER IF EXISTS trg_review_likes ON review_likes")
    op.drop_table("review_likes")
    op.drop_table("reviews")
    op.drop_table("books")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS book_status")
    op.execute("DROP TYPE IF EXISTS sync_action")
    op.execute("DROP TYPE IF EXISTS sync_entity")
