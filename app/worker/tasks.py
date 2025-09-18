import json
import logging
import os
from contextlib import contextmanager
from typing import Generator, Dict, Any, List, Optional

import redis
from pydub import AudioSegment
from pathlib import Path
from sqlmodel import Session, select

from app.core.config import settings
from app.db.base import engine
from app.db.models import MeetingJob, Transcription, DiarizedTranscript
from app.processing.diarization import SpeakerDiarization
from app.processing.enrollment import SpeakerEnrollment
from app.processing.mapper import map_speaker_to_text
from app.processing.transcription import Transcriber
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

_transcriber_service: Optional[Transcriber] = None
_diarization_service: Optional[SpeakerDiarization] = None
_enrollment_service: Optional[SpeakerEnrollment] = None

def get_transcriber() -> Transcriber:
    global _transcriber_service
    if _transcriber_service is None:
        logger.info("Initializing Transcriber service for this process...")
        _transcriber_service = Transcriber()
    return _transcriber_service

def get_diarizer() -> SpeakerDiarization:
    global _diarization_service
    if _diarization_service is None:
        logger.info("Initializing SpeakerDiarization service for this process...")
        _diarization_service = SpeakerDiarization()
    return _diarization_service


def get_enrollment_service() -> SpeakerEnrollment:
    global _enrollment_service
    if _enrollment_service is None:
        logger.info("Initializing SpeakerEnrollment service for this process...")
        _enrollment_service = SpeakerEnrollment()
    return _enrollment_service

# ===================================================================
#   Database and Pub/Sub Helpers
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
    
    request_id_for_publish = None
    try:
        transcriber = get_transcriber()
        sentence_transcript, word_transcript = transcriber.transcribe(audio_path, language)
        
        if not sentence_transcript:
            raise ValueError("Transcription resulted in an empty output.")

        with db_session() as session:
            job = session.get(MeetingJob, job_id)
            if not job:
                logger.error(f"[Job ID: {job_id}] Job not found in database. Aborting.")
                return

            request_id_for_publish = job.request_id

            new_transcription = Transcription(
                meeting_job_id=job.id,
                language=language,
                transcript_data=word_transcript 
            )
            session.add(new_transcription)
            

            job.status = "transcription_complete"
            job.error_message = None
            session.add(job)


        update_payload = {
            "status": "transcription_complete",
            "plain_transcript": sentence_transcript 
        }
        publish_job_update(request_id_for_publish, update_payload)
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

    request_id_for_publish = None
    try:
        diarizer = get_diarizer()
        enrollment_svc = get_enrollment_service()
        with db_session() as session:
            job = session.get(MeetingJob, job_id)
            if not job:
                logger.error(f"[Job ID: {job_id}] Job not found. Aborting diarization.")
                return

            request_id_for_publish = job.request_id

            transcription_entry = session.exec(
                select(Transcription).where(
                    Transcription.meeting_job_id == job.id,
                    Transcription.language == job.language
                )
            ).first()

            if not transcription_entry or not transcription_entry.transcript_data:
                raise FileNotFoundError("Could not find a valid source transcript for diarization.")

            word_level_transcript = transcription_entry.transcript_data

        enrolled_profiles = enrollment_svc.get_all_enrolled_profiles_for_diarization()

        speaker_segments = diarizer.diarize(audio_path, enrolled_profiles, enrollment_svc)
        if not speaker_segments:
            raise ValueError("Diarization process did not produce any speaker segments.")

        diarized_transcript = map_speaker_to_text(speaker_segments, word_level_transcript)

        with db_session() as session:
            job = session.get(MeetingJob, job_id)

            new_diarized_entry = DiarizedTranscript(
                meeting_job_id=job.id,
                transcript_data=diarized_transcript,
                is_edited=False
            )
            session.add(new_diarized_entry)
            
            job.status = "completed"
            job.error_message = None
            session.add(job)


        update_payload = {
            "status": "completed",
            "diarized_transcript": diarized_transcript
        }
        publish_job_update(request_id_for_publish, update_payload)
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


@celery_app.task(bind=True, name="enroll_speaker_task")
def enroll_speaker_task(self, user_ad: str, audio_sample_paths: List[str], metadata: dict):
    """
    Celery task to perform speaker enrollment in the background.
    This task is routed to the 'gpu_tasks' queue.
    """
    logger.info(f"[Task] Starting enrollment for new speaker: '{user_ad}'")
    try:
        enrollment_svc = get_enrollment_service()
        enrollment_svc.enroll_new_speaker(
            user_ad=user_ad,
            audio_sample_paths=audio_sample_paths,
            metadata=metadata
        )
        logger.info(f"[Task] Enrollment successful for '{user_ad}'.")
        # Can replace with websocket later
        
    except Exception as e:
        logger.error(f"[Task] Enrollment failed for '{user_ad}': {e}", exc_info=True)
        raise


@celery_app.task(bind=True, name="assemble_audio_task")
def assemble_audio_task(self, request_id: str, language: str):
    """Assembles audio chunks and then triggers the transcription task."""
    logger.info(f"[Task ID: {self.request.id}] Assembling chunks for job '{request_id}'...")
    session_dir = Path(settings.SHARED_AUDIO_PATH) / request_id
    job_id_for_next_task, full_audio_path_for_next_task = None, None
    try:
        chunk_files = sorted(
            [f for f in session_dir.iterdir() if f.is_file() and f.stem.split('_')[-1].isdigit()],
            key=lambda f: int(f.stem.split('_')[-1])
        )
        if not chunk_files: raise FileNotFoundError("No valid chunk files found.")

        combined_audio = AudioSegment.empty()
        for chunk_path in chunk_files:
            combined_audio += AudioSegment.from_file(chunk_path)

        final_audio = combined_audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        with db_session() as session:
            job = session.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
            if not job: return
            
            final_filename = f"{Path(job.original_filename).stem}_full.wav"
            full_audio_path = session_dir / final_filename
            final_audio.export(full_audio_path, format="wav")
            
            job.status = "transcribing"
            session.add(job)
            job_id_for_next_task, full_audio_path_for_next_task = job.id, str(full_audio_path)
            
        publish_job_update(request_id, {"status": "transcribing"})
        for chunk_path in chunk_files: os.remove(chunk_path)
            
        celery_app.send_task("run_transcription_task", args=[job_id_for_next_task, full_audio_path_for_next_task, language])
        logger.info(f"Assembly complete. Dispatched transcription task for job '{request_id}'.")
    except Exception as e:
        logger.error(f"Assembly task failed for job '{request_id}': {e}", exc_info=True)
        with db_session() as session:
            job = session.exec(select(MeetingJob).where(MeetingJob.request_id == request_id)).first()
            if job:
                job.status = "failed"
                job.error_message = f"Audio Assembly Failed: {str(e)}"
                session.add(job)
                publish_job_update(job.request_id, {"status": "failed", "error_message": job.error_message})
        raise


@celery_app.task(bind=True, name="add_samples_task")
def add_samples_task(self, user_ad: str, new_audio_paths: List[str]):
    """
    Adds new voice samples to an existing speaker profile.
    This is a GPU task because it requires generating new embeddings.
    """
    logger.info(f"[Task] Adding {len(new_audio_paths)} samples to speaker '{user_ad}'")
    try:
        enrollment_svc = get_enrollment_service()
        enrollment_svc.add_samples_to_profile(user_ad, new_audio_paths)
        logger.info(f"[Task] Successfully added samples to '{user_ad}'.")
    except Exception as e:
        logger.error(f"[Task] Failed to add samples for '{user_ad}': {e}", exc_info=True)
        raise

@celery_app.task(bind=True, name="update_metadata_task")
def update_metadata_task(self, user_ad: str, new_metadata: dict):
    """
    Updates a speaker's metadata. This is a CPU task because it only involves
    a database write and regenerating search terms (a string operation).
    """
    logger.info(f"[Task] Updating metadata for speaker '{user_ad}'")
    try:
        enrollment_svc = get_enrollment_service()
        enrollment_svc.update_metadata(user_ad, new_metadata)
        logger.info(f"[Task] Successfully updated metadata for '{user_ad}'.")
    except Exception as e:
        logger.error(f"Failed to update metadata for '{user_ad}': {e}", exc_info=True)
        raise
