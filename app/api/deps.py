import logging
from typing import Generator

from fastapi import Depends, HTTPException, status, Form, Query, Path
from sqlmodel import Session, select

from app.db.base import engine
from app.db.models import User, MeetingJob, Transcription

logger = logging.getLogger(__name__)

# ===================================================================
#   DATABASE DEPENDENCY
# ===================================================================

def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency that creates and yields a new database session for
    each request, ensuring the session is always closed.
    """
    with Session(engine) as session:
        yield session

# ===================================================================
#   USER IDENTIFICATION DEPENDENCIES
# ===================================================================

def get_or_create_user(
    session: Session = Depends(get_db_session),
    username: str = Form(..., description="The username of the user making the request.")
) -> User:
    """
    Dependency to get a user from a POST form. If the user does not exist,
    a new one is created.
    """
    user = session.exec(select(User).where(User.username == username)).first()
    if not user:
        logger.info(f"User '{username}' not found from form. Creating new user record.")
        user = User(username=username, display_name=username)
        session.add(user)
        session.commit()
        session.refresh(user)
    return user

def get_or_create_user_from_query(
    session: Session = Depends(get_db_session),
    username: str = Query(..., description="The username of the user making the request.")
) -> User:
    """
    Dependency to get a user from a URL query parameter (for GET requests).
    If the user does not exist, a new one is created.
    """
    user = session.exec(select(User).where(User.username == username)).first()
    if not user:
        logger.info(f"User '{username}' not found from query. Creating new user record.")
        user = User(username=username, display_name=username)
        session.add(user)
        session.commit()
        session.refresh(user)
    return user

# ===================================================================
#   JOB OWNERSHIP & STATE VERIFICATION DEPENDENCIES
# ===================================================================

# --- CORE OWNERSHIP VERIFIERS ---

def get_owned_job_from_path(
    request_id: str = Path(..., description="The unique request_id of the meeting job."),
    current_user: User = Depends(get_or_create_user_from_query),
    db: Session = Depends(get_db_session),
) -> MeetingJob:
    """
    Core dependency for GET/PUT/DELETE requests.
    1. Finds a job by its request_id from the URL path.
    2. Verifies the current user (from query param) is the job's owner.
    Raises 404 if not found, 403 if ownership fails.
    """

    job = db.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting job with request_id '{request_id}' not found."
        )
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to access this resource."
        )
    return job

def get_owned_job_from_form(
    requestId: str = Form(..., description="The unique request_id of the meeting job."),
    current_user: User = Depends(get_or_create_user),
    db: Session = Depends(get_db_session),
) -> MeetingJob:
    """
    Core dependency for POST requests with form data.
    1. Finds a job by its request_id from the form body.
    2. Verifies the current user (from form body) is the job's owner.
    Raises 404 if not found, 403 if ownership fails.
    """
    job = db.exec(select(MeetingJob).where(MeetingJob.request_id == requestId)).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting job with request_id '{requestId}' not found."
        )
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to access this resource."
        )
    return job

# --- STATE-AWARE DEPENDENCIES (Build on top of core verifiers) ---

def get_job_ready_for_diarization(
    job: MeetingJob = Depends(get_owned_job_from_path)
) -> MeetingJob:
    """
    Verifies ownership and ensures the job is in the 'transcription_complete' state,
    making it eligible for diarization to be triggered.
    """
    if job.status != "transcription_complete":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Diarization can only be started when job status is 'transcription_complete'. Current status is '{job.status}'."
        )
    return job

def get_job_with_completed_diarization(
    job: MeetingJob = Depends(get_owned_job_from_path)
) -> MeetingJob:
    """
    Verifies ownership and ensures the job has a final, speaker-separated transcript.
    This gates features like 'summarize by speaker'.
    """
    if job.status != "completed" or not job.diarized_transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This feature requires a completed speaker-separated transcript. Please run diarization first."
        )
    return job

def get_job_with_any_transcript(
    job: MeetingJob = Depends(get_owned_job_from_path),
    db: Session = Depends(get_db_session)
) -> MeetingJob:
    """
    Verifies ownership and that at least one transcript (in any language) exists for the job.
    This gates features like 'summarize by topic', chat, and editing.
    """
    # Check if a transcription record exists for the job's current language
    transcription = db.exec(
        select(Transcription).where(
            Transcription.meeting_job_id == job.id,
            Transcription.language == job.language
        )
    ).first()

    if not transcription:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This feature requires a transcript for the current language. Please ensure transcription is complete."
        )
    return job

def get_cancellable_job(
    job: MeetingJob = Depends(get_owned_job_from_path)
) -> MeetingJob:
    """
    Verifies ownership and that the job is in a state where cancellation is permitted
    (i.e., before the main processing has kicked off).
    """
    cancellable_states = ["uploading", "assembling"]
    if job.status not in cancellable_states:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Meeting cannot be cancelled. It is already in the '{job.status}' state."
        )
    return job