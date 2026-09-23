"""ORM models for dataset metadata and version lineage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from backend.auth import LOCAL_USER_ID as LOCAL_OWNER_ID
from backend.database import Base

# JSONB on Postgres; plain JSON elsewhere (e.g. local SQLite).
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Dataset(Base):
    """Root dataset record uploaded into the Soft Sensor Toolbox."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    # Supabase auth user id, or "local" when AUTH_REQUIRED is off.
    owner_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default=LOCAL_OWNER_ID, index=True
    )
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    original_schema: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    # Set in Python too: SQLite's now() has one-second resolution, and eviction
    # of the oldest dataset must not tie with an upload made the same second.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    versions: Mapped[list[DatasetVersion]] = relationship(
        "DatasetVersion",
        back_populates="dataset",
        cascade="all, delete-orphan",
        foreign_keys="DatasetVersion.dataset_id",
        order_by="DatasetVersion.created_at",
    )


class DatasetVersion(Base):
    """Immutable snapshot of a dataset after a transformation or upload."""

    __tablename__ = "dataset_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("dataset_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    action_performed: Mapped[str] = mapped_column(Text, nullable=False)
    parameters_used: Mapped[dict[str, Any]] = mapped_column(
        JSONType,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    dataset: Mapped[Dataset] = relationship(
        "Dataset",
        back_populates="versions",
        foreign_keys=[dataset_id],
    )
    parent_version: Mapped[Optional[DatasetVersion]] = relationship(
        "DatasetVersion",
        remote_side=[id],
        foreign_keys=[parent_version_id],
    )


class TrainedModel(Base):
    """A saved soft-sensor artifact and the metadata needed to list it without unpickling."""

    __tablename__ = "trained_models"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # model_id, algorithm, target/feature columns, version ids, r2_score, rmse, trained_at.
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    artifact_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class ProjectState(Base):
    """The UI session (chat, plots, selections, active step) kept until New project."""

    __tablename__ = "project_state"

    owner_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
