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
    JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
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

    job: Mapped[Job] = relationship(back_populates="segments")

    __table_args__ = (
        Index("ix_segments_job_index", "job_id", "index", unique=True),
    )
