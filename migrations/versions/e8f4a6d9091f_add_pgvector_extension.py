"""add_pgvector_extension

Revision ID: e8f4a6d9091f
Revises: 20cd882f5ce8
Create Date: 2026-08-25 18:15:58.404100

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8f4a6d9091f'
down_revision: Union[str, Sequence[str], None] = '20cd882f5ce8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    pass


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
    pass
