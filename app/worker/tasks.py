import json
import logging
from contextlib import contextmanager
from typing import Generator, Dict, Any

import redis
from sqlmodel import Session, select

from app.core.config import settings
from app.db.base import engine
from app.db.models import MeetingJob, Transcription
from app.processing.diarization import SpeakerDiarization
from app.processing.enrollment import SpeakerEnrollment
from app.processing.mapper import map_speaker_to_text
from app.processing.transcription import Transcriber
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# ===================================================================
#   Singleton Service Instantiation
# ===================================================================
# These heavy objects are created ONCE when the Celery worker process starts.
# This is highly efficient as it avoids reloading models for every task.

transcriber_service: Transcriber = None
diarization_service: SpeakerDiarization = None
enrollment_service: SpeakerEnrollment = None

def get_transcriber() -> Transcriber:
    """Initializes and returns a singleton Transcriber instance."""
    global transcriber_service
    if transcriber_service is None:
        logger.info("Initializing Transcriber service for this worker...")
        transcriber_service = Transcriber()
    return transcriber_service

def get_diarizer() -> SpeakerDiarization:
    """Initializes and returns a singleton SpeakerDiarization instance."""
    global diarization_service
    if diarization_service is None:
        logger.info("Initializing SpeakerDiarization service for this worker...")
        diarization_service = SpeakerDiarization()
    return diarization_service

def get_enrollment_service() -> SpeakerEnrollment:
    """Initializes and returns a singleton SpeakerEnrollment instance."""
    global enrollment_service
    if enrollment_service is None:
        logger.info("Initializing SpeakerEnrollment service for this worker...")
        enrollment_service = SpeakerEnrollment()
    return enrollment_service

# ===================================================================
#   Helper Functions for State Management and Communication
# ===================================================================

@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Provides a transactional database session for tasks."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def publish_job_update(request_id: str, update_data: Dict[str, Any]):
    """Publishes a job status update to Redis Pub/Sub for real-time frontend updates."""
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        payload = json.dumps({"request_id": request_id, "data": update_data})
        r.publish("job_updates", payload)
        logger.info(f"Published update for job '{request_id}': {update_data.get('status')}")
    except Exception as e:
        logger.error(f"Failed to publish Redis update for job '{request_id}': {e}", exc_info=True)

# ===================================================================
#   Celery Task Definitions
# ===================================================================

@celery_app.task(bind=True, name="run_transcription_task")
def run_transcription_task(self, job_id: int, audio_path: str, language: str):
    """
    Celery task to perform transcription on an audio file.
    This task is routed to the 'gpu_tasks' queue.
    """
    logger.info(f"[Job ID: {job_id}] Starting transcription for audio '{audio_path}' in '{language}'.")
    
    try:
        # Step 1: Initialize the transcription service (loads model if not already loaded)
        transcriber = get_transcriber()

        # Step 2: Perform the transcription
        sentence_transcript, word_transcript = transcriber.transcribe(audio_path, language)
        
        if not sentence_transcript:
            raise ValueError("Transcription resulted in an empty output.")

        # Step 3: Update the database
        with db_session() as session:
            job = session.get(MeetingJob, job_id)
            if not job:
                logger.error(f"[Job ID: {job_id}] Job not found in database. Aborting.")
                return

            # Create a new Transcription record linked to the job
            new_transcription = Transcription(
                meeting_job_id=job.id,
                language=language,
                transcript_data=word_transcript # Store the detailed word-level data
            )
            session.add(new_transcription)
            
            # Update the main job status
            job.status = "transcription_complete"
            job.error_message = None
            session.add(job)

        # Step 4: Notify the frontend via WebSocket
        update_payload = {
            "status": "transcription_complete",
            "plain_transcript": sentence_transcript # Send the user-friendly sentence data
        }
        publish_job_update(job.request_id, update_payload)
        logger.info(f"[Job ID: {job_id}] Transcription completed and saved successfully.")

    except Exception as e:
        logger.error(f"[Job ID: {job_id}] Transcription failed: {e}", exc_info=True)
        with db_session() as session:
            job = session.get(MeetingJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = f"Transcription Error: {str(e)}"
                session.add(job)
                publish_job_update(job.request_id, {"status": "failed", "error_message": job.error_message})
        raise

@celery_app.task(bind=True, name="run_diarization_task")
def run_diarization_task(self, job_id: int, audio_path: str):
    """
    Celery task to perform diarization and map speakers to an existing transcript.
    This task is also routed to the 'gpu_tasks' queue.
    """
    logger.info(f"[Job ID: {job_id}] Starting diarization for audio '{audio_path}'.")

    try:
        # Step 1: Initialize services
        diarizer = get_diarizer()
        enrollment_svc = get_enrollment_service()

        # Step 2: Fetch necessary data
        with db_session() as session:
            job = session.get(MeetingJob, job_id)
            if not job:
                logger.error(f"[Job ID: {job_id}] Job not found. Aborting diarization.")
                return

            # Find the correct word-level transcript for the job's active language
            transcription_entry = session.exec(
                select(Transcription).where(
                    Transcription.meeting_job_id == job.id,
                    Transcription.language == job.language
                )
            ).first()

            if not transcription_entry or not transcription_entry.transcript_data:
                raise FileNotFoundError("Could not find a valid source transcript for diarization.")

            word_level_transcript = transcription_entry.transcript_data

        # Fetch all known speaker profiles for matching
        enrolled_profiles = enrollment_svc.get_all_enrolled_profiles_for_diarization()

        # Step 3: Perform the diarization to get speaker segments
        speaker_segments = diarizer.diarize(audio_path, enrolled_profiles, enrollment_svc)
        if not speaker_segments:
            raise ValueError("Diarization process did not produce any speaker segments.")

        # Step 4: Map the speaker segments to the word-level transcript
        diarized_transcript = map_speaker_to_text(speaker_segments, word_level_transcript)

        # Step 5: Update the database
        with db_session() as session:
            job = session.get(MeetingJob, job_id)
            # You would create a new `DiarizedTranscript` model here and save `diarized_transcript` data
            # For now, let's assume we save it back to the job for simplicity
            new_diarized_entry = DiarizedTranscript(
                meeting_job_id=job.id,
                transcript_data=diarized_transcript,
                is_edited=False # It's fresh from the AI, so not edited yet
            )
            session.add(new_diarized_entry)
            
            job.status = "completed"
            job.error_message = None
            session.add(job)

        # Step 6: Notify the frontend
        update_payload = {
            "status": "completed",
            "diarized_transcript": diarized_transcript
        }
        publish_job_update(job.request_id, update_payload)
        logger.info(f"[Job ID: {job_id}] Diarization and mapping completed successfully.")

    except Exception as e:
        logger.error(f"[Job ID: {job_id}] Diarization failed: {e}", exc_info=True)
        with db_session() as session:
            job = session.get(MeetingJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = f"Diarization Error: {str(e)}"
                session.add(job)
                publish_job_update(job.request_id, {"status": "failed", "error_message": job.error_message})
        raise
