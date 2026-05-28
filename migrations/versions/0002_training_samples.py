"""Add training_samples table (Feedback Loop / Active Learning)

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-25
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_samples",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("predict_id", sa.String(36), nullable=True),
        sa.Column("image_path", sa.Text, nullable=True),
        sa.Column("ai_title", sa.String(500), nullable=True),
        sa.Column("ai_author", sa.String(500), nullable=True),
        sa.Column("ai_confidence", sa.String(20), nullable=True),
        sa.Column("ai_method", sa.String(20), nullable=True),
        sa.Column("correct_title", sa.String(500), nullable=False),
        sa.Column("correct_author", sa.String(500), nullable=False),
        sa.Column("was_corrected", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("needs_review", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("used_in_training", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.text("now()"),
        ),
        sa.Column(
            "book_id", UUID(as_uuid=True),
            sa.ForeignKey("books.id", ondelete="SET NULL"), nullable=True,
        ),
    )
    op.create_index("idx_training_predict_id", "training_samples", ["predict_id"],
                    postgresql_where=sa.text("predict_id IS NOT NULL"))
    op.create_index("idx_training_pending", "training_samples", ["used_in_training"],
                    postgresql_where=sa.text("used_in_training = false"))
    op.create_index("idx_training_corrected", "training_samples", ["was_corrected"])


def downgrade() -> None:
    op.drop_table("training_samples")
