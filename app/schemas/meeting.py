from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ===================================================================
#   Core Data Structure Models
# ===================================================================

class PlainSegment(BaseModel):
    """
    The direct output from the transcription model (Whisper).
    """
    id: Optional[int] = Field(None, description="A unique identifier for the segment.")
    text: str = Field(..., description="The transcribed text of the segment.")
    start_time: float = Field(..., description="Start time of the segment in seconds.")
    end_time: float = Field(..., description="End time of the segment in seconds.")

class DiarizedSegment(BaseModel):
    """
    Represents a single segment of a transcript AFTER speaker separation.
    It includes a speaker label.
    """
    id: Optional[int] = Field(None, description="A unique identifier for the segment.")
    speaker: str = Field(..., description="The identified speaker for this segment (e.g., 'LinhPT' or 'Unknown_Speaker_1').")
    text: str = Field(..., description="The transcribed text of the segment.")
    start_time: float = Field(..., description="Start time of the segment in seconds.")
    end_time: float = Field(..., description="End time of the segment in seconds.")

# ===================================================================
#   API Request Body Models
# ===================================================================

class MeetingInfoUpdateRequest(BaseModel):
    """
    Schema for updating the metadata of a meeting. All fields are optional
    so the user can update one or more fields at a time.
    """
    bbh_name: Optional[str] = None
    meeting_type: Optional[str] = None
    meeting_host: Optional[str] = None

class LanguageChangeRequest(BaseModel):
    """
    Schema for the specific action of changing a meeting's language,
    which may trigger a new transcription task.
    """
    language: str = Field(..., description="The new language code for the meeting (e.g., 'vi', 'en').")

class PlainTranscriptUpdateRequest(BaseModel):
    """
    Schema for submitting user edits to the plain (non-diarized) transcript.
    """
    segments: List[PlainSegment]

class DiarizedTranscriptUpdateRequest(BaseModel):
    """
    Schema for submitting user edits to the speaker-separated (diarized) transcript.
    """
    segments: List[DiarizedSegment]

class SummaryRequest(BaseModel):
    """
    Schema for requesting a new summary. Specifies the type of summary to generate.
    """
    summary_type: str = Field(..., description="The type of summary to generate (e.g., 'topic', 'speaker', 'action_items', 'decision_log').")

class ChatRequest(BaseModel):
    requestId: str
    username: str
    message: str

# ===================================================================
#   API Response Body Models
# ===================================================================

class MeetingStatusResponse(BaseModel):
    """
    A comprehensive model representing the full state of a meeting job.
    """
    request_id: str = Field(..., description="The unique identifier for the meeting job.")
    status: str = Field(..., description="The current processing status of the job (e.g., 'uploading', 'transcribing', 'transcription_complete', 'diarizing', 'completed').")
    
    # Meeting Metadata
    bbh_name: str
    meeting_type: str
    meeting_host: str
    language: str = Field(..., description="The currently active language of the meeting.")
    
    # Transcript Data
    plain_transcript: Optional[List[PlainSegment]] = Field(None, description="The transcript before speaker separation. Available when status is 'transcription_complete' or later.")
    diarized_transcript: Optional[List[DiarizedSegment]] = Field(None, description="The final transcript with speaker labels. Available when status is 'completed'.")
    
    error_message: Optional[str] = None

class MeetingJobResponseWrapper(BaseModel):
    """Standard wrapper for API responses for consistency."""
    status: int = 200
    message: str = "Success"
    data: MeetingStatusResponse

class SummaryResponse(BaseModel):
    """
    Response model for a successfully generated summary.
    """
    request_id: str
    summary_type: str = Field(..., description="The type of the summary provided.")
    summary_content: str = Field(..., description="The generated summary content in Markdown or plain text.")

class ChatResponse(BaseModel):
    success: int = 0
    response: str

class DownloadLinkResponse(BaseModel):
    """
    Response model providing a temporary, secure link to download an audio file.
    """
    request_id: str
    download_url: str
    expires_at: datetime
