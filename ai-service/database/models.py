"""
Database Models
"""

from enum import Enum


class JobStatus(Enum):
    """Job status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class BatchJob:
    """Represents a batch job in database"""
    pass


class VideoTask:
    """Represents a video task in database"""
    pass
