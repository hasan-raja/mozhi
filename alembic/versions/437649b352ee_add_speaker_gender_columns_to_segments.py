"""add speaker gender columns to segments

Revision ID: 437649b352ee
Revises: 6581c84b56af
Create Date: 2026-09-01 18:16:06.264634

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '437649b352ee'
down_revision: str | None = '6581c84b56af'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add speaker and gender columns to segments table
    op.add_column('segments', sa.Column('speaker', sa.String(32), nullable=True))
    op.add_column('segments', sa.Column('gender', sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column('segments', 'gender')
    op.drop_column('segments', 'speaker')
