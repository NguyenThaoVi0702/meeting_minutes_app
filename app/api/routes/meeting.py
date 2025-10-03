import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from datetime import timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import quote

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status,
    WebSocket, WebSocketDisconnect
)
from fastapi.responses import FileResponse, StreamingResponse
from pydub import AudioSegment, exceptions as pydub_exceptions
from sqlmodel import Session, select

from app.api.deps import (
    get_cancellable_job, get_db_session, get_job_ready_for_diarization,
    get_job_with_any_transcript, get_job_with_completed_diarization,
    get_or_create_user, get_owned_job_from_path
)
from app.core.config import settings
from app.db.base import engine
from app.db.models import ChatHistory, MeetingJob, Summary, Transcription, User
from app.schemas.meeting import (
    ChatRequest, ChatResponse, LanguageChangeRequest, MeetingInfoUpdateRequest,
    MeetingJobResponseWrapper, MeetingStatusResponse, PlainSegment,
    PlainTranscriptUpdateRequest, SummaryRequest, SummaryResponse
)
from app.services.ai_service import ai_service
from app.services.document_generator import generate_templated_document, generate_docx_from_markdown
from app.services.websocket_manager import websocket_manager
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter()



# ===================================================================
#   Helper Functions & Core Workflow
# ===================================================================

async def _generate_and_save_summary(db: Session, job: MeetingJob, summary_type: str) -> Summary:
    """
    Internal helper to generate a summary, save it to the DB, and return the new object.
    This contains the core generation logic, preventing code duplication.
    """
    logger.info(f"Generating a new '{summary_type}' summary for job '{job.request_id}'.")
    

    if summary_type == "speaker":
        if not job.diarized_transcript:
            raise HTTPException(status_code=400, detail="A 'speaker' summary requires diarization.")
        transcript_source = job.diarized_transcript.transcript_data
        source_text = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in transcript_source])
    else:
        transcription_entry = db.exec(select(Transcription).where(Transcription.meeting_job_id == job.id, Transcription.language == job.language)).first()
        if not transcription_entry or not transcription_entry.transcript_data:
            raise HTTPException(status_code=400, detail="A summary requires a completed transcript.")
        transcript_source = transcription_entry.transcript_data
        source_text = "\n".join([seg['text'] for seg in transcript_source])

   
    try:
 
        meeting_info = {
            "bbh_name": job.bbh_name,
            "meeting_type": job.meeting_type,
            "meeting_host": job.meeting_host,
            "start_time": "N/A",
            "end_time": "N/A",
            "ngay": "N/A",
            "thang": "N/A",
            "nam": "N/A",
            "meeting_members_str": ", ".join(job.meeting_members) if job.meeting_members else "Không xác định",
        }
        
  
        if summary_type in ["summary_bbh_hdqt", "summary_nghi_quyet"]:
            local_tz = ZoneInfo("Asia/Ho_Chi_Minh")
            start_time_local = job.upload_started_at.replace(tzinfo=timezone.utc).astimezone(local_tz) if job.upload_started_at else None
            end_time_local = job.upload_finished_at.replace(tzinfo=timezone.utc).astimezone(local_tz) if job.upload_finished_at else None

            if start_time_local:
                meeting_info["start_time"] = start_time_local.strftime('%H:%M')
                meeting_info["ngay"] = start_time_local.strftime('%d')
                meeting_info["thang"] = start_time_local.strftime('%m')
                meeting_info["nam"] = start_time_local.strftime('%Y') 
            
            if end_time_local:
                meeting_info["end_time"] = end_time_local.strftime('%H:%M')

            context_header = (
                f"**THÔNG TIN BỐI CẢNH CUỘC HỌP:**\n"
                f"- Ngày họp: {start_time_local.strftime('%d/%m/%Y') if start_time_local else 'N/A'}\n"
                f"- Giờ bắt đầu: {meeting_info['start_time']}\n"
                f"- Giờ kết thúc: {meeting_info['end_time']}\n\n"
                f"**NỘI DUNG BIÊN BẢN (TRANSCRIPT):**\n"
            )
            source_text = context_header + source_text


        summary_content = await ai_service.get_response(
            task=summary_type,
            user_message=source_text,
            context={"meeting_info": meeting_info}
        )
    except Exception as e:
        logger.error(f"AI service call failed: {e}", exc_info=True) # Add more detail for debugging
        raise HTTPException(status_code=502, detail=f"Failed to get response from AI service: {e}")


    new_summary = Summary(
        meeting_job_id=job.id,
        summary_type=summary_type,
        summary_content=summary_content
    )
    db.add(new_summary)
    db.commit()
    db.refresh(new_summary)
    logger.info(f"Successfully generated and saved new '{summary_type}' summary.")
    return new_summary

def _parse_ai_json(json_string: str) -> Optional[Dict]:
    """Safely parses a JSON string that might be wrapped in markdown."""
    try:
        cleaned_string = re.sub(r'```json\s*|\s*```', '', json_string, flags=re.DOTALL).strip()
        return json.loads(cleaned_string)
    except (json.JSONDecodeError, TypeError):
        logger.error(f"Failed to decode AI JSON response: {json_string}")
        return None

def _format_seconds_to_hms(seconds: float) -> str:
    """Converts a float number of seconds to an HH:MM:SS string."""
    if not isinstance(seconds, (int, float)):
        return "00:00:00"
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    seconds = s % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _format_job_status(job: MeetingJob, db: Session) -> dict:
    """Packages a MeetingJob object into the standard API response schema."""
    plain_transcript_data = None
    diarized_transcript_data = None 

    transcription_entry = db.exec(
        select(Transcription).where(
            Transcription.meeting_job_id == job.id,
            Transcription.language == job.language
        )
    ).first()

    if transcription_entry and transcription_entry.transcript_data:
        raw_segments = transcription_entry.transcript_data
        plain_transcript_data = [
            {
                **seg,
                "start_time": _format_seconds_to_hms(seg.get("start_time", 0)),
                "end_time": _format_seconds_to_hms(seg.get("end_time", 0)),
            } for seg in raw_segments
        ]

    if job.diarized_transcript and job.diarized_transcript.transcript_data:
        raw_diarized = job.diarized_transcript.transcript_data
        diarized_transcript_data = [
            {
                **seg,
                "start_time": _format_seconds_to_hms(seg.get("start_time", 0)),
                "end_time": _format_seconds_to_hms(seg.get("end_time", 0)),
            } for seg in raw_diarized
        ]


    response = MeetingStatusResponse(
        request_id=job.request_id,
        status=job.status,
        bbh_name=job.bbh_name,
        meeting_type=job.meeting_type,
        meeting_host=job.meeting_host,
        language=job.language,
        plain_transcript=[PlainSegment(**seg) for seg in plain_transcript_data] if plain_transcript_data else None,
        diarized_transcript=[DiarizedSegment(**seg) for seg in diarized_transcript_data] if diarized_transcript_data else None,
        error_message=job.error_message
    )
    return response.model_dump()





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
    meetingMembers: str = Form("[]", description="A JSON string of a list of member names."),
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

    members_list = []
    try:
        parsed_members = json.loads(meetingMembers)
        if isinstance(parsed_members, list):
            members_list = parsed_members
        else:
            raise ValueError("meetingMembers must be a JSON array (a list of strings).")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid format for meetingMembers: {e}")

    job = MeetingJob(
        request_id=requestId,
        user_id=current_user.id,
        language=language,
        original_filename=filename,
        bbh_name=bbhName,
        meeting_type=Type,
        meeting_host=Host,
        meeting_members=members_list,
        status="uploading"
    )
    session.add(job)
    session.commit()
    return {"status": 201, "message": "Meeting initialized. Ready for chunk uploads."}


@router.post("/upload-file-chunk", status_code=status.HTTP_202_ACCEPTED, summary="Upload a single audio chunk")
async def upload_file_chunk(
    #background_tasks: BackgroundTasks,
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

    if not job.upload_started_at:
        job.upload_started_at = datetime.utcnow()
        logger.info(f"First chunk received for '{requestId}'. Recording start time.")

    session_dir = Path(settings.SHARED_AUDIO_PATH) / requestId
    chunk_path = session_dir / FileData.filename
    
    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(FileData.file, buffer)
    except IOError as e: 
        raise HTTPException(status_code=500, detail=f"Failed to save chunk file: {e}")

    if isLastChunk:
        job.status = "assembling"
        job.upload_finished_at = datetime.utcnow()
        logger.info(f"Last chunk received for '{requestId}'. Recording end time.")

        session.add(job)
        session.commit()
        logger.info(f"Last chunk received for '{requestId}'. Triggering background assembly and transcription.")
        
        celery_app.send_task("assemble_audio_task", args=[requestId, job.language], queue="gpu_tasks")
        await websocket_manager.broadcast_to_job(requestId, {"status": "assembling"})
    else:
        session.add(job)
        session.commit()

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

    celery_app.send_task("run_diarization_task", args=[job.id, str(audio_file_path)],  queue="gpu_tasks")

    await websocket_manager.broadcast_to_job(job.request_id, {"status": "diarizing"})
    
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
    db.add(job)
    formatted_data = _format_job_status(job, db)
    return MeetingJobResponseWrapper(data=formatted_data)


@router.websocket("/ws/{request_id}")
async def websocket_endpoint(websocket: WebSocket, request_id: str):
    """
    Establishes a WebSocket connection for receiving real-time updates
    about a meeting job's status.
    """
    await websocket.accept()
    
    if request_id not in websocket_manager.active_connections:
        websocket_manager.active_connections[request_id] = []
    websocket_manager.active_connections[request_id].append(websocket)
    logger.info(f"WebSocket connected for request_id '{request_id}'.")

    try:
        with Session(engine) as session:
            job = session.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
            if job:
                initial_status = _format_job_status(job, session)
                await websocket.send_json(initial_status)

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for request_id '{request_id}'.")
    finally:
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
    db.add(job)
    update_dict = update_data.model_dump(exclude_unset=True)
    if not update_dict:
        raise HTTPException(status_code=400, detail="No update data provided.")
        
    for key, value in update_dict.items():
        setattr(job, key, value)
    
    db.commit()
    db.refresh(job)

    updated_status = _format_job_status(job, db)
    await websocket_manager.broadcast_to_job(job.request_id, updated_status)

    return MeetingJobResponseWrapper(message="Meeting info updated successfully.", data=updated_status)


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
    db.add(job)
    new_language = language_request.language
    if job.language == new_language:
        return MeetingJobResponseWrapper(data=_format_job_status(job, db), message="Language is already set to the requested one.")

    cached_transcript = db.exec(
        select(Transcription).where(
            Transcription.meeting_job_id == job.id,
            Transcription.language == new_language
        )
    ).first()

    job.language = new_language
    
    if cached_transcript:
        logger.info(f"Found cached transcript for language '{new_language}' for job '{job.request_id}'.")
        if job.diarized_transcript:
             db.delete(job.diarized_transcript)
             job.diarized_transcript = None
             db.flush()
        job.status = "transcription_complete"
    else:
        logger.info(f"No cached transcript for '{new_language}'. Triggering new transcription task for job '{job.request_id}'.")
        audio_file_path = Path(settings.SHARED_AUDIO_PATH) / job.request_id / f"{Path(job.original_filename).stem}_full.wav"
        if not audio_file_path.exists():
            raise HTTPException(status_code=404, detail="Assembled audio file not found. Cannot re-transcribe.")
        
        job.status = "transcribing"
        celery_app.send_task("run_transcription_task", args=[job.id, str(audio_file_path), new_language],  queue="gpu_tasks")

    db.commit()
    db.refresh(job)

    updated_status = _format_job_status(job, db)
    await websocket_manager.broadcast_to_job(job.request_id, updated_status)

    return MeetingJobResponseWrapper(message=f"Language changed to '{new_language}'.", data=updated_status)


@router.put(
    "/{request_id}/transcript/plain",
    response_model=MeetingJobResponseWrapper,
    summary="Update the plain (non-diarized) transcript"
)
async def update_plain_transcript(
    update_request: PlainTranscriptUpdateRequest,
    db: Session = Depends(get_db_session),
    job: MeetingJob = Depends(get_owned_job_from_path)
):
    """
    Overwrites the current plain transcript with user-provided edits.
    **IMPORTANT**: Submitting a new transcript will PERMANENTLY DELETE any existing diarized transcript
    and all previously generated summaries for this meeting, as they will be
    based on outdated information. The job status will revert to
    'transcription_complete', requiring diarization to be run again.
    """
    db.add(job)
    logger.info(f"Received request to update plain transcript for job '{job.request_id}'.")

    transcription_entry = db.exec(
        select(Transcription).where(
            Transcription.meeting_job_id == job.id,
            Transcription.language == job.language
        )
    ).first()

    if not transcription_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active transcript found for language '{job.language}'. Cannot update."
        )

    # Invalidate and delete all downstream data that depends on this transcript
    logger.warning(f"Invalidating downstream data for job '{job.request_id}' due to transcript edit.")


    if job.diarized_transcript:
        logger.info(f"Deleting stale diarized transcript for job '{job.request_id}'.")
        db.delete(job.diarized_transcript)
        job.diarized_transcript = None
        # Revert status 
        job.status = "transcription_complete"
        db.flush()

    # Delete all associated summaries
    if job.summaries:
        logger.info(f"Deleting {len(job.summaries)} stale summaries for job '{job.request_id}'.")
        for summary in job.summaries:
            db.delete(summary)
    
    # Delete chat history as it may refer to old text. 
    if job.chat_history:
        logger.info(f"Deleting {len(job.chat_history)} stale chat history entries for job '{job.request_id}'.")
        for chat_entry in job.chat_history:
            db.delete(chat_entry)

    new_transcript_data = [seg.model_dump() for seg in update_request.segments]
    transcription_entry.transcript_data = new_transcript_data
    transcription_entry.is_edited = True # Mark this transcript as user-modified

    db.add(transcription_entry) 
    db.commit()
    db.refresh(job)

    logger.info(f"Successfully updated transcript for job '{job.request_id}'.")
    updated_status = _format_job_status(job, db)
    await websocket_manager.broadcast_to_job(job.request_id, updated_status)

    return MeetingJobResponseWrapper(
        message="Transcript updated successfully. All dependent data like diarization and summaries have been cleared.",
        data=updated_status
    )


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

    db.delete(job)
    db.commit()

    await websocket_manager.broadcast_to_job(job.request_id, {"status": "cancelled", "message": "The meeting has been cancelled."})

    return {"status": 200, "message": "Meeting successfully cancelled."}


# ===================================================================
#   Analysis, Chat, and Download Endpoints
# ===================================================================

@router.post(
    "/{request_id}/summary",
    response_model=SummaryResponse,
    summary="Get, or generate and save, a meeting summary"
)
async def get_or_generate_summary( 
    summary_request: SummaryRequest,
    db: Session = Depends(get_db_session),
    job: MeetingJob = Depends(get_owned_job_from_path)
):
    """
    Retrieves a summary if it already exists in the database.
    If not, it generates the summary, saves it permanently, and then returns it.
    """
    db.add(job)
    summary_type = summary_request.summary_type
    
    existing_summary = db.exec(
        select(Summary).where(
            Summary.meeting_job_id == job.id,
            Summary.summary_type == summary_type
        )
    ).first()

    if existing_summary:
        summary_to_return = existing_summary
    else:
        summary_to_return = await _generate_and_save_summary(db, job, summary_type)

    return SummaryResponse(
        request_id=job.request_id,
        summary_type=summary_to_return.summary_type,
        summary_content=summary_to_return.summary_content
    )



@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat about the meeting content"
)
async def chat_with_meeting(
    chat_request: ChatRequest,
    db: Session = Depends(get_db_session)
):
    job = db.exec(select(MeetingJob).where(MeetingJob.request_id == chat_request.requestId)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Meeting job not found.")

    final_response_to_user = "Xin lỗi, tôi chưa hiểu ý của bạn. Bạn có thể diễn đạt khác được không?"

    # --- STAGE 1: INTENT ANALYSIS ---
    try:
        intent_json_str = await ai_service.get_response(task="intent_analysis", user_message=chat_request.message)
        intent_data = _parse_ai_json(intent_json_str)

        if not intent_data:
            raise ValueError("AI did not return valid JSON for intent analysis.")

        intent = intent_data.get("intent")
        entity = intent_data.get("entity")
        edit_instruction = intent_data.get("edit_instruction") or chat_request.message

    except Exception as e:
        logger.error(f"Failed during Stage 1 (Intent Analysis): {e}")
        intent = 'ask_question' # Default to simple question if analysis fails

    # --- STAGE 2: BACKEND ORCHESTRATION ---

    # --- PATH 1: USER WANTS TO EDIT A SUMMARY ---
    if intent == 'edit_summary':
        if not entity:
            # AMBIGUITY: Ask the user to clarify
            final_response_to_user = "Bạn muốn sửa loại tóm tắt nào? (ví dụ: theo chủ đề, theo người nói, các công việc cần làm...)"
        else:
            # State Check: See if the summary exists
            summary_record = db.exec(
                select(Summary).where(Summary.meeting_job_id == job.id, Summary.summary_type == entity)
            ).first()

            if not summary_record:
                # Summary does not exist: Ask user to generate it first
                final_response_to_user = f"Biên bản họp theo '{entity}' chưa được tạo. Vui lòng nhấn nút tương ứng để tạo tóm tắt trước khi bạn có thể chỉnh sửa."
            else:
                # HAPPY PATH: Summary exists, so we can edit it.
                logger.info(f"Performing summary edit for type '{entity}' on job '{job.request_id}'.")
                
                # Construct the detailed prompt for the Stage 2 generation call
                edit_context = (
                    f"--- EXISTING '{entity.upper()}' SUMMARY ---\n"
                    f"{summary_record.summary_content}\n\n"
                    f"--- USER'S EDIT INSTRUCTION ---\n"
                    f"{edit_instruction}"
                )
                
                # Using the chat prompt with the update rules
                new_summary_content_raw = await ai_service.get_response(
                    task="chat", 
                    user_message=edit_context
                )
                
                # Parse the response to get the clean summary content
                update_pattern = r'\[UPDATE:(\w+)\]\s*(.*)'
                match = re.match(update_pattern, new_summary_content_raw, re.DOTALL)
                
                if match:
                    new_summary_content = match.group(2).strip()
                    summary_record.summary_content = new_summary_content
                    db.add(summary_record)
                    final_response_to_user = new_summary_content
                else:
                    # If the AI fails to follow the format, return its raw response
                    final_response_to_user = new_summary_content_raw

    # --- PATH 2: USER IS ASKING A GENERAL QUESTION ---
    elif intent == 'ask_question':
        logger.info(f"Answering a general question for job '{job.request_id}'.")
        transcript_entry = db.exec(select(Transcription).where(Transcription.meeting_job_id == job.id, Transcription.language == job.language)).first()
        transcript_text = "\n".join([seg['text'] for seg in transcript_entry.transcript_data]) if transcript_entry else ""
        
        summaries = db.exec(select(Summary).where(Summary.meeting_job_id == job.id)).all()
        summary_texts = [f"--- SUMMARY ({s.summary_type.upper()}) ---\n{s.summary_content}" for s in summaries]

        chat_history_db = db.exec(select(ChatHistory).where(ChatHistory.meeting_job_id == job.id).order_by(ChatHistory.created_at.desc()).limit(settings.LIMIT_TURN * 2)).all()
        chat_history_formatted = [{"role": entry.role, "content": entry.message} for entry in reversed(chat_history_db)]
        
        full_context_for_llm = (f"--- MEETING TRANSCRIPT ---\n{transcript_text}\n\n" + "\n\n".join(summary_texts))
        
        final_response_to_user = await ai_service.get_response(
            task="chat",
            user_message=f"**User Question:**\n{chat_request.message}\n\n**Meeting Context:**\n{full_context_for_llm}",
            context={"history": chat_history_formatted}
        )

    # --- PATH 3: GENERAL CHIT-CHAT (OR FALLBACK) ---
    else:
        final_response_to_user = "Cảm ơn bạn. Tôi có thể giúp gì khác cho cuộc họp này không?"

    # --- SAVE CONVERSATION TO HISTORY AND COMMIT ALL CHANGES ---
    db.add(ChatHistory(meeting_job_id=job.id, role="user", message=chat_request.message))
    db.add(ChatHistory(meeting_job_id=job.id, role="assistant", message=final_response_to_user))
    db.commit()

    return ChatResponse(response=final_response_to_user)


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

    safe_filename = f"Meeting_Audio_{job.bbh_name.replace(' ', '_')}.wav"
    
    return FileResponse(
        path=audio_file_path,
        media_type='audio/wav',
        filename=safe_filename
    )


@router.get("/{request_id}/download/document", summary="Download any summary as a DOCX document")
async def download_summary_document(
    job: MeetingJob = Depends(get_owned_job_from_path),
    summary_type: str = Query(..., description="The type of summary to download."),
    db: Session = Depends(get_db_session)
):
    """
    Downloads the latest version of any given summary type as a .docx file.
    It fetches the content from the database. If the summary hasn't been generated yet,
    it will return an error.
    """
    # 1. Fetch the required summary from the database
    summary = db.exec(
        select(Summary).where(
            Summary.meeting_job_id == job.id,
            Summary.summary_type == summary_type
        )
    ).first()

    if not summary:
        logger.warning(f"Summary '{summary_type}' not found for download. Generating now...")
        summary = await _generate_and_save_summary(db, job, summary_type)

    try:
        if summary_type in ["summary_bbh_hdqt", "summary_nghi_quyet"]:
            document_buffer = generate_templated_document(summary_type.replace("summary_", ""), summary.summary_content)
        else:
            document_buffer = generate_docx_from_markdown(summary.summary_content)
    except Exception as e:
        logger.error(f"Error during document generation for type '{summary_type}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate the document: {e}")

    safe_filename_part = job.bbh_name.replace(' ', '_').replace('/', '_')
    encoded_filename = quote(f"{summary_type}_{safe_filename_part}.docx")
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"}
    
    return StreamingResponse(
        document_buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers
    )
