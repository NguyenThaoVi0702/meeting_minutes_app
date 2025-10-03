import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlmodel import SQLModel, Field, Relationship

# ===================================================================
#   User and Logging Models (Largely Unchanged)
# ===================================================================

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    display_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    meeting_jobs: List["MeetingJob"] = Relationship(back_populates="owner")
    action_logs: List["SpeakerActionLog"] = Relationship(back_populates="submitter")

class SpeakerActionLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: str = Field(index=True)
    submitter_id: int = Field(foreign_key="user.id")
    action_type: str = Field(index=True)
    target_user_ad: Optional[str] = Field(default=None, index=True)
    device_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    payload: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    status: str
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    submitter: User = Relationship(back_populates="action_logs")

# ===================================================================
#   Core Meeting & Data Models (Significantly Enhanced)
# ===================================================================

class MeetingJob(SQLModel, table=True):
    """
    The central table for a meeting session. It tracks metadata, status,
    and links to all related data like transcripts, summaries, etc.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: str = Field(unique=True, index=True)
    user_id: int = Field(foreign_key="user.id")
    
    original_filename: str
    bbh_name: str
    meeting_type: str
    meeting_host: str
    meeting_members: Optional[List[str]] = Field(default=None, sa_column=Column(JSONB))
    language: str = Field(default="vi", description="The currently active language for this meeting.")
    upload_started_at: Optional[datetime] = Field(default=None, description="Timestamp of when the first audio chunk was received.")
    upload_finished_at: Optional[datetime] = Field(default=None, description="Timestamp of when the last audio chunk was received.")

    status: str = Field(default="uploading", index=True,
                        description="Tracks the current state of the job, e.g., 'uploading', 'transcribing', 'transcription_complete', 'diarizing', 'completed'.")
    diarization_job_id: Optional[str] = Field(default=None, index=True)
    error_message: Optional[str] = Field(default=None)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)})

    owner: User = Relationship(back_populates="meeting_jobs")
    

    transcriptions: List["Transcription"] = Relationship(
        back_populates="meeting_job", sa_relationship_kwargs={"cascade": "all, delete"}
    )
    

    diarized_transcript: Optional["DiarizedTranscript"] = Relationship(
        back_populates="meeting_job", sa_relationship_kwargs={"cascade": "all, delete", "uselist": False}
    )
    
    summaries: List["Summary"] = Relationship(
        back_populates="meeting_job", sa_relationship_kwargs={"cascade": "all, delete"}
    )
    
    chat_history: List["ChatHistory"] = Relationship(
        back_populates="meeting_job", sa_relationship_kwargs={"cascade": "all, delete"}
    )


class Transcription(SQLModel, table=True):
    """
    Stores a language-specific transcript for a meeting. A meeting can have
    multiple entries here if the user switches languages.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_job_id: int = Field(foreign_key="meetingjob.id")
    language: str = Field(index=True, description="The language code of this transcript (e.g., 'vi', 'en').")

    transcript_data: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSONB), description="Sentence level transcript for user display")
    word_level_data: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSONB), description="Word level transcript for internal processing")

    is_edited: bool = Field(default=False)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)})

    meeting_job: MeetingJob = Relationship(back_populates="transcriptions")


class DiarizedTranscript(SQLModel, table=True):
    """
    Stores the final, speaker-separated transcript after diarization has run.
    A meeting will have only one entry here. Its presence indicates that speaker-
    dependent features can be enabled.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_job_id: int = Field(foreign_key="meetingjob.id", unique=True)

    transcript_data: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSONB))

    is_edited: bool = Field(default=False)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)})

    meeting_job: MeetingJob = Relationship(back_populates="diarized_transcript")


class Summary(SQLModel, table=True):
    """
    Stores a generated summary for a meeting. This new design allows for multiple
    types of summaries for a single meeting.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_job_id: int = Field(foreign_key="meetingjob.id")

    summary_type: str = Field(index=True)

    summary_content: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)})

    meeting_job: MeetingJob = Relationship(back_populates="summaries")


class ChatHistory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_job_id: int = Field(foreign_key="meetingjob.id")
    role: str  
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    meeting_job: "MeetingJob" = Relationship(back_populates="chat_history")
