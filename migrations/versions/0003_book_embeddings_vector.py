"""Add book_embeddings table with pgvector (Vector DB / CLIP)

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-25

ШААРДЛАГА:
  PostgreSQL-д pgvector өргөтгөл суусан байх шаардлагатай.

  Локал суулгалт (Windows PostgreSQL 17):
    https://github.com/pgvector/pgvector#windows

  Docker ашиглаж буй бол pgvector багтсан зураглал хэрэглэнэ:
    image: pgvector/pgvector:pg16   # docker-compose.yml-д солих

  Суусны дараа migration ажиллуулна:
    alembic upgrade 0003
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # pgvector өргөтгөл идэвхжүүлэх
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    # book_embeddings хүснэгт үүсгэх (raw SQL — vector тип pgvector-с ирдэг)
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS book_embeddings (
            book_id     UUID PRIMARY KEY
                        REFERENCES books(id) ON DELETE CASCADE,
            embedding   vector(512) NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """))

    # IVFFlat индекс — cosine ижилтэлийн хайлтыг хурдасгана
    # (Датасет 1000+-аас дээш болохоор ашигтай)
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_book_embeddings_cosine
        ON book_embeddings
        USING ivfflat (embedding vector_cosine_ops)
    """))


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS book_embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")
