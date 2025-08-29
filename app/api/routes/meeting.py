import os
import json
import shutil
import logging
from pathlib import Path

from fastapi import (
    APIRouter, Depends, Form, File, UploadFile, HTTPException, BackgroundTasks,
    WebSocket, WebSocketDisconnect, status
)
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from pydub import AudioSegment, exceptions as pydub_exceptions

from app.api.deps import (
    get_db_session, get_or_create_user, get_owned_job_from_path,
    get_job_ready_for_diarization, get_cancellable_job
)
from app.core.config import settings
from app.db.models import MeetingJob, Transcription
from app.schemas.meeting import (
    MeetingStatusResponse, MeetingJobResponseWrapper, PlainSegment,
    MeetingInfoUpdateRequest, LanguageChangeRequest
)
from app.services.websocket_manager import websocket_manager
from app.worker.tasks import run_transcription_task, run_diarization_task

logger = logging.getLogger(__name__)
router = APIRouter()

# ===================================================================
#   Helper Functions
# ===================================================================

def _format_job_status(job: MeetingJob, db: Session) -> dict:
    """
    Packages a MeetingJob object and its related data into the
    standard API response schema.
    """
    plain_transcript_data = None
    
    # Find the plain transcript for the job's currently active language
    transcription_entry = db.exec(
        select(Transcription).where(
            Transcription.meeting_job_id == job.id,
            Transcription.language == job.language
        )
    ).first()

    if transcription_entry and transcription_entry.transcript_data:
        # Assuming transcript_data is a list of dicts that matches PlainSegment
        plain_transcript_data = [PlainSegment(**seg) for seg in transcription_entry.transcript_data]

    response = MeetingStatusResponse(
        request_id=job.request_id,
        status=job.status,
        bbh_name=job.bbh_name,
        meeting_type=job.meeting_type,
        meeting_host=job.meeting_host,
        language=job.language,
        plain_transcript=plain_transcript_data,
        diarized_transcript=job.diarized_transcript.transcript_data if job.diarized_transcript else None,
        error_message=job.error_message
    )
    return response.model_dump()

async def assemble_and_transcribe(request_id: str, language: str):
    """
    Background task to assemble audio chunks and trigger the Celery
    transcription task.
    """
    logger.info(f"[BG Task] Assembling chunks for job '{request_id}'...")
    session_dir = Path(settings.SHARED_AUDIO_PATH) / request_id
    
    try:
        # Find all numbered chunk files
        chunk_files = sorted(
            [f for f in session_dir.iterdir() if f.is_file() and f.stem.split('_')[-1].isdigit()],
            key=lambda f: int(f.stem.split('_')[-1])
        )
        if not chunk_files:
            raise FileNotFoundError("No valid chunk files found for assembly.")

        # Combine audio chunks
        combined_audio = AudioSegment.empty()
        for chunk_path in chunk_files:
            combined_audio += AudioSegment.from_file(chunk_path)

        # Standardize and export the final audio file
        final_audio = combined_audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        
        with Session(engine) as session:
            job = session.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
            if not job: return
            final_filename = f"{Path(job.original_filename).stem}_full.wav"
            full_audio_path = session_dir / final_filename
            final_audio.export(full_audio_path, format="wav")
            
            # Update job status and notify clients
            job.status = "transcribing"
            session.add(job)
            session.commit()
            await websocket_manager.broadcast_to_job(request_id, {"status": "transcribing"})

        # Clean up chunk files
        for chunk_path in chunk_files:
            os.remove(chunk_path)
            
        # Trigger the Celery task for the heavy lifting
        run_transcription_task.delay(job.id, str(full_audio_path), language)
        logger.info(f"Successfully assembled audio and dispatched transcription task for job '{request_id}'.")

    except (FileNotFoundError, pydub_exceptions.CouldntDecodeError, Exception) as e:
        logger.error(f"Failed during background assembly for job '{request_id}': {e}", exc_info=True)
        with Session(engine) as session:
            job = session.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
            if job:
                job.status = "failed"
                job.error_message = f"Audio Assembly Failed: {e}"
                session.add(job)
                session.commit()
                await websocket_manager.broadcast_to_job(request_id, {"status": "failed", "error_message": job.error_message})

# ===================================================================
#   Core Meeting Workflow Endpoints
# ===================================================================

@router.post("/start-bbh", status_code=status.HTTP_201_CREATED, summary="Initialize a new meeting session")
async def start_bbh(
    session: Session = Depends(get_db_session),
    current_user: User = Depends(get_or_create_user),
    requestId: str = Form(...),
    language: str = Form("vi"),
    filename: str = Form(...),
    bbhName: str = Form(...),
    Type: str = Form(...),
    Host: str = Form(...),
):
    """
    Initializes a meeting job record in the database and creates a
    dedicated directory for storing incoming audio chunks.
    """
    logger.info(f"Initializing job '{requestId}' by user '{current_user.username}'.")
    if session.exec(select(MeetingJob).where(MeetingJob.request_id == requestId)).first():
        raise HTTPException(status_code=409, detail=f"Meeting job with requestId '{requestId}' already exists.")

    session_dir = Path(settings.SHARED_AUDIO_PATH) / requestId
    os.makedirs(session_dir, exist_ok=True)

    job = MeetingJob(
        request_id=requestId,
        user_id=current_user.id,
        language=language,
        original_filename=filename,
        bbh_name=bbhName,
        meeting_type=Type,
        meeting_host=Host,
        status="uploading"
    )
    session.add(job)
    session.commit()
    return {"status": 201, "message": "Meeting initialized. Ready for chunk uploads."}


@router.post("/upload-file-chunk", status_code=status.HTTP_202_ACCEPTED, summary="Upload a single audio chunk")
async def upload_file_chunk(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db_session),
    requestId: str = Form(...),
    isLastChunk: bool = Form(...),
    FileData: UploadFile = File(...),
):
    """
    Receives an audio chunk, saves it, and if it's the last chunk,
    triggers a background task to assemble the audio and start transcription.
    """
    job = session.exec(select(MeetingJob).where(MeetingJob.request_id == requestId)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Meeting job not found.")
    if job.status != "uploading":
        raise HTTPException(status_code=400, detail=f"Cannot upload chunks when job status is '{job.status}'.")

    session_dir = Path(settings.SHARED_AUDIO_PATH) / requestId
    chunk_path = session_dir / FileData.filename
    
    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(FileData.file, buffer)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chunk file: {e}")

    if isLastChunk:
        job.status = "assembling"
        session.add(job)
        session.commit()
        logger.info(f"Last chunk received for '{requestId}'. Triggering background assembly and transcription.")
        await websocket_manager.broadcast_to_job(requestId, {"status": "assembling"})
        background_tasks.add_task(assemble_and_transcribe, requestId, job.language)

    return {"status": 202, "message": f"Chunk '{FileData.filename}' accepted."}


@router.post("/{request_id}/diarize", status_code=status.HTTP_202_ACCEPTED, summary="Trigger speaker diarization")
async def diarize_meeting(
    db: Session = Depends(get_db_session),
    job: MeetingJob = Depends(get_job_ready_for_diarization)
):
    """
    Triggers the speaker diarization and mapping process for a meeting that
    has a completed transcription.
    """
    audio_file_path = Path(settings.SHARED_AUDIO_PATH) / job.request_id / f"{Path(job.original_filename).stem}_full.wav"
    if not audio_file_path.exists():
        raise HTTPException(status_code=404, detail="Assembled audio file not found. Please re-upload.")

    job.status = "diarizing"
    db.add(job)
    db.commit()

    await websocket_manager.broadcast_to_job(job.request_id, {"status": "diarizing"})
    
    run_diarization_task.delay(job.id, str(audio_file_path))
    
    return {"status": 202, "message": "Diarization process started."}

# ===================================================================
#   Real-time Status and Management Endpoints
# ===================================================================

@router.get("/{request_id}/status", response_model=MeetingJobResponseWrapper, summary="Get current meeting status")
async def get_meeting_status(
    db: Session = Depends(get_db_session),
    job: MeetingJob = Depends(get_owned_job_from_path)
):
    """
    Retrieves the complete current status of a meeting job, including any
    available transcripts. Ideal for initial page loads.
    """
    formatted_data = _format_job_status(job, db)
    return MeetingJobResponseWrapper(data=formatted_data)


@router.websocket("/ws/{request_id}")
async def websocket_endpoint(websocket: WebSocket, request_id: str, db: Session = Depends(get_db_session)):
    """
    Establishes a WebSocket connection for receiving real-time updates
    about a meeting job's status.
    """
    # For WebSocket, we cannot use standard Depends, so we manually verify ownership.
    # In a real app, you'd pass a token and validate it. For now, we trust the client.
    await websocket_manager.connect(websocket, request_id)
    try:
        # Send the current status immediately upon connection
        job = db.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
        if job:
            initial_status = _format_job_status(job, db)
            await websocket.send_json(initial_status)
        
        # Keep the connection alive to receive broadcasted updates
        while True:
            await websocket.receive_text() # Keep connection open
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, request_id)


@router.patch("/{request_id}/info", response_model=MeetingJobResponseWrapper, summary="Update meeting metadata")
async def update_meeting_info(
    update_data: MeetingInfoUpdateRequest,
    db: Session = Depends(get_db_session),
    job: MeetingJob = Depends(get_owned_job_from_path)
):
    """
    Updates editable meeting metadata (name, type, host) at any time.
    """
    update_dict = update_data.model_dump(exclude_unset=True)
    if not update_dict:
        raise HTTPException(status_code=400, detail="No update data provided.")
        
    for key, value in update_dict.items():
        setattr(job, key, value)
    
    db.add(job)
    db.commit()
    db.refresh(job)

    # Broadcast the change to all connected clients
    updated_status = _format_job_status(job, db)
    await websocket_manager.broadcast_to_job(job.request_id, updated_status)

    return MeetingJobResponseWrapper(message="Meeting info updated successfully.", data=updated_status)

# --- End of Part 1 ---

# ... (Continuing from Part 1) ...

# ===================================================================
#   Editing, Language, and Cancellation Endpoints
# ===================================================================

@router.post("/{request_id}/language", response_model=MeetingJobResponseWrapper, summary="Change meeting language")
async def change_meeting_language(
    language_request: LanguageChangeRequest,
    db: Session = Depends(get_db_session),
    job: MeetingJob = Depends(get_owned_job_from_path)
):
    """
    Changes the active language of the meeting. If a transcript for the new
    language doesn't exist, it triggers a new transcription task.
    """
    new_language = language_request.language
    if job.language == new_language:
        return MeetingJobResponseWrapper(data=_format_job_status(job, db), message="Language is already set to the requested one.")

    # Check if a transcript for this language already exists
    cached_transcript = db.exec(
        select(Transcription).where(
            Transcription.meeting_job_id == job.id,
            Transcription.language == new_language
        )
    ).first()

    job.language = new_language
    
    if cached_transcript:
        logger.info(f"Found cached transcript for language '{new_language}' for job '{job.request_id}'.")
        # Diarized transcript is language-independent, but we clear it to avoid confusion
        if job.diarized_transcript:
             db.delete(job.diarized_transcript)
        job.status = "transcription_complete"
    else:
        logger.info(f"No cached transcript for '{new_language}'. Triggering new transcription task for job '{job.request_id}'.")
        audio_file_path = Path(settings.SHARED_AUDIO_PATH) / job.request_id / f"{Path(job.original_filename).stem}_full.wav"
        if not audio_file_path.exists():
            raise HTTPException(status_code=404, detail="Assembled audio file not found. Cannot re-transcribe.")
        
        job.status = "transcribing"
        run_transcription_task.delay(job.id, str(audio_file_path), new_language)

    db.add(job)
    db.commit()
    db.refresh(job)

    # Broadcast the updated status
    updated_status = _format_job_status(job, db)
    await websocket_manager.broadcast_to_job(job.request_id, updated_status)

    return MeetingJobResponseWrapper(message=f"Language changed to '{new_language}'.", data=updated_status)


@router.put("/{request_id}/transcript", status_code=status.HTTP_200_OK, summary="Update plain transcript")
async def update_plain_transcript(
    # A new dependency will be needed to get the specific transcription object
    # For now, let's put logic here.
    # TODO: Refactor into a dependency
    request_id: str = Path(...),
    # ...
):
    """
    Allows a user to submit their edits to the plain (non-diarized) transcript.
    (Implementation placeholder - requires a dedicated dependency and schema).
    """
    # 1. Get the job and verify ownership.
    # 2. Find the correct Transcription record based on job.language.
    # 3. Update the `transcript_data` with the user's submitted segments.
    # 4. Set `is_edited = True`.
    # 5. Commit changes and broadcast the update.
    raise HTTPException(status_code=501, detail="Endpoint not yet implemented.")


@router.delete("/{request_id}/cancel", status_code=status.HTTP_200_OK, summary="Cancel an ongoing meeting")
async def cancel_meeting(
    db: Session = Depends(get_db_session),
    job: MeetingJob = Depends(get_cancellable_job)
):
    """
    Allows a user to cancel a meeting that is still in the 'uploading' or
    'assembling' phase. This deletes the job record and all associated files.
    """
    logger.info(f"Received cancellation request for job '{job.request_id}'.")
    session_dir = Path(settings.SHARED_AUDIO_PATH) / job.request_id
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
            logger.info(f"Removed temporary directory: {session_dir}")
        except OSError as e:
            logger.error(f"Error removing directory {session_dir} during cancellation: {e}")
            # Non-fatal, we still want to delete the DB record.

    db.delete(job)
    db.commit()
    
    # Notify connected clients that the job has been cancelled and is now gone
    await websocket_manager.broadcast_to_job(job.request_id, {"status": "cancelled", "message": "The meeting has been cancelled."})

    return {"status": 200, "message": "Meeting successfully cancelled."}


# ===================================================================
#   Analysis, Chat, and Download Endpoints
# ===================================================================

@router.post("/{request_id}/summary", summary="Generate a meeting summary")
async def generate_summary(
    # ... Implementation for generating summaries based on type
):
    """
    Generates a summary for the meeting based on the requested type.
    - 'topic', 'action_items', 'decision_log' require a completed transcript.
    - 'speaker' requires a completed diarized transcript.
    (Implementation placeholder).
    """
    raise HTTPException(status_code=501, detail="Endpoint not yet implemented.")


@router.post("/chat", summary="Chat about the meeting content")
async def chat_with_meeting(
    # ... Implementation for chat functionality
):
    """
    Handles conversational queries about a completed meeting.
    (Implementation placeholder).
    """
    raise HTTPException(status_code=501, detail="Endpoint not yet implemented.")


@router.get("/{request_id}/download/audio", summary="Download the original audio file")
async def download_audio_file(
    job: MeetingJob = Depends(get_owned_job_from_path)
):
    """
    Provides a direct download of the fully assembled meeting audio file.
    """
    logger.info(f"Request to download audio for job '{job.request_id}'.")
    audio_file_path = Path(settings.SHARED_AUDIO_PATH) / job.request_id / f"{Path(job.original_filename).stem}_full.wav"

    if not audio_file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assembled audio file not found. It may not have been processed yet."
        )

    # Sanitize filename for the Content-Disposition header
    safe_filename = f"Meeting_Audio_{job.bbh_name.replace(' ', '_')}.wav"
    
    return FileResponse(
        path=audio_file_path,
        media_type='audio/wav',
        filename=safe_filename
    )