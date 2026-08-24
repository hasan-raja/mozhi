"""Pipeline stage registry — single source of truth for queues and ordering.

Adding a stage = add one Stage entry; celery routes, worker command, and the
future WebSocket progress mapping all read from here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    queue: str


STAGES: list[Stage] = [
    Stage("extract", "extract"),
    Stage("vad", "vad"),
    Stage("asr", "asr"),
    Stage("translate", "translate"),
    Stage("tts", "tts"),
    Stage("qc", "qc"),
    Stage("stitch", "stitch"),
]

STAGE_NAMES = [s.name for s in STAGES]
