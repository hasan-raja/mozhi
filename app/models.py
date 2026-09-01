"""Domain models — Job state machine first.

Design decisions:
- State transitions enforced at DB level where possible (CHECK constraints),
  app-level guard as the friendly error path.
- Index on (status, created_at) exists for the future stuck-job reaper scan:
  "WHERE status IN (running,) AND heartbeat < now() - ttl".
"""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db_base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    CANCELLED = "cancelled"


# Legal transitions — anything else raises InvalidTransition at app level.
VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.RUNNING, JobStatus.CANCELLED},
    # RUNNING → DEAD_LETTERED: the reaper direct-DLQs jobs whose attempts are
    # exhausted (no point bouncing through FAILED just to fail again).
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.DEAD_LETTERED,
        JobStatus.CANCELLED,
    },
    JobStatus.FAILED: {JobStatus.PENDING, JobStatus.DEAD_LETTERED},  # retry or DLQ
    JobStatus.DEAD_LETTERED: set(),
    JobStatus.COMPLETED: set(),
    JobStatus.CANCELLED: set(),
}

TERMINAL_STATUSES = {JobStatus.COMPLETED, JobStatus.DEAD_LETTERED,
                     JobStatus.CANCELLED}
# FAILED is NOT terminal: it's the retry gateway (failed → pending → running).


class InvalidTransition(Exception):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          default=uuid.uuid4)
    source_lang: Mapped[str] = mapped_column(String(10))
    target_lang: Mapped[str] = mapped_column(String(10))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", native_enum=True),
        default=JobStatus.PENDING,
    )
    attempt: Mapped[int] = mapped_column(default=0)
    max_attempts: Mapped[int] = mapped_column(default=3)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                        onupdate=utcnow())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Heartbeat for the reaper: running tasks touch this periodically.
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    failure_reason: Mapped[str | None] = mapped_column(String(1000))

    segments: Mapped[list["Segment"]] = relationship(back_populates="job",
                                                     cascade="all, delete-orphan")

    __table_args__ = (
        # Reaper + dashboard scans filter by status and sort by time.
        Index("ix_jobs_status_created", "status", "created_at"),
    )

    def transition(self, to: JobStatus) -> None:
        """App-level guard; DB CHECK is the last line of defense."""
        allowed = VALID_TRANSITIONS[self.status]
        if to not in allowed:
            raise InvalidTransition(f"{self.status.value} → {to.value} not allowed")
        if self.status in TERMINAL_STATUSES:
            raise InvalidTransition(f"{self.status.value} is terminal")
        if to == JobStatus.RUNNING:
            self.started_at = self.started_at or utcnow()
        if to in TERMINAL_STATUSES:
            self.finished_at = utcnow()
        self.status = to


class Segment(Base):
    """One utterance of a job — produced by VAD/ASR, consumed by TTS/stitch."""

    __tablename__ = "segments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    index: Mapped[int]  # order within the job
    start_ms: Mapped[int]
    end_ms: Mapped[int]

    source_text: Mapped[str | None]
    translated_text: Mapped[str | None]
    audio_path: Mapped[str | None]  # blob layout: jobs/{job_id}/tts/{seg}.wav

    qc_score: Mapped[float | None]
    qc_passed: Mapped[bool | None]

    # Speaker + gender (Day 4 - gender voice mapping)
    speaker: Mapped[str | None] = mapped_column(String(32))
    gender: Mapped[str | None] = mapped_column(String(16))  # male|female|unknown

    job: Mapped[Job] = relationship(back_populates="segments")

    __table_args__ = (
        Index("ix_segments_job_index", "job_id", "index", unique=True),
    )


class StageCompletion(Base):
    """Idempotency ledger — at-least-once delivery means a task MAY run twice;
    this table guarantees completing a stage twice is impossible.

    The UNIQUE constraint is the enforcement mechanism: racing workers cannot
    both insert. DB-level > app-level because it holds across processes.
    """

    __tablename__ = "stage_completions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                   server_default=func.now())

    __table_args__ = (
        Index("ux_stage_completions_job_stage", "job_id", "stage", unique=True),
    )


class Asset(Base):
    """A piece of media: uploaded source or generated artifact.

    Two roles:
    - role="source": the user's upload (video/audio in)
    - role="<stage>": per-stage outputs (extract wav, tts wavs, final mp4)
    Paths follow the blob layout jobs/{job_id}/{stage}/{name} — resumable and
    debuggable; the DB row is metadata, the file lives in blob storage.
    """

    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # source|extract|tts|final...
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(100))
    size_bytes: Mapped[int | None]
    duration_ms: Mapped[int | None]
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now())

    __table_args__ = (
        Index("ix_assets_job_role", "job_id", "role"),
    )


class UsageRecord(Base):
    """Metering row — one per billable API action.

    Foundation of mozhi-sdk metering: every stage run appends usage so billing
    is a query, never an afterthought. Immutable by convention (no updates).
    """

    __tablename__ = "usage_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          default=uuid.uuid4)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("jobs.id"))
    api_key_id: Mapped[str | None] = mapped_column(String(64))  # tenant identity (Day 5 SDK)
    engine: Mapped[str] = mapped_column(String(32), nullable=False)  # local|sarvam|openrouter|mock
    operation: Mapped[str] = mapped_column(String(32), nullable=False)  # asr|translate|tts
    quantity: Mapped[float] = mapped_column(default=0.0)  # seconds audio / chars / requests
    unit: Mapped[str] = mapped_column(String(16), nullable=False)  # audio_sec|chars|requests
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now())

    __table_args__ = (
        Index("ix_usage_apikey_created", "api_key_id", "created_at"),
        Index("ix_usage_job", "job_id"),
    )
