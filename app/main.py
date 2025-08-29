import logging
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis
from sqlmodel import Session, select, SQLModel

from app.api.routes import meeting, speaker
from app.core.config import settings, ensure_directories_exist
from app.db.base import engine
from app.db.models import MeetingJob
from app.services.websocket_manager import websocket_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================================================================
#   Background Tasks (Redis Listener & Cleanup)
# ===================================================================

async def redis_listener():
    """
    Listens to the 'job_updates' Redis channel and broadcasts messages
    to connected WebSocket clients. Runs for the entire application lifetime.
    """
    try:
        r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        async with r.pubsub() as pubsub:
            await pubsub.subscribe("job_updates")
            logger.info("Redis listener subscribed to 'job_updates' channel.")
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=None)
                if message:
                    try:
                        logger.debug(f"Received Redis message: {message['data']}")
                        payload = json.loads(message["data"])
                        request_id = payload.get("request_id")
                        data_to_send = payload.get("data")
                        if request_id and data_to_send:
                            await websocket_manager.broadcast_to_job(request_id, data_to_send)
                    except json.JSONDecodeError:
                        logger.error(f"Could not decode Redis message: {message['data']}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}", exc_info=True)
    except asyncio.CancelledError:
        logger.info("Redis listener task cancelled.")
    except Exception as e:
        logger.critical(f"Redis listener failed critically: {e}", exc_info=True)


async def cleanup_stale_jobs():
    """
    Periodically cleans up old jobs that were started but never completed,
    preventing orphaned files and database entries.
    """
    while True:
        await asyncio.sleep(3600 * 6) # Run every 6 hours
        logger.info("Running cleanup task for stale, incomplete jobs...")
        # Incomplete jobs older than 2 days
        threshold = datetime.utcnow() - timedelta(days=2)
        stale_states = ["uploading", "assembling", "transcribing", "diarizing"]

        try:
            with Session(engine) as session:
                stale_jobs = session.exec(
                    select(MeetingJob).where(
                        MeetingJob.status.in_(stale_states),
                        MeetingJob.created_at < threshold
                    )
                ).all()

                if not stale_jobs:
                    logger.info("No stale jobs found to clean up.")
                    continue

                for job in stale_jobs:
                    logger.warning(f"Found stale job '{job.request_id}'. Setting status to 'failed'.")
                    job.status = "failed"
                    job.error_message = "Job timed out and was cleaned up automatically."
                    session.add(job)
                session.commit()
        except Exception as e:
            logger.error(f"Error during stale job cleanup: {e}", exc_info=True)


# ===================================================================
#   Application Lifecycle (Startup & Shutdown)
# ===================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    """
    # --- Startup ---
    logger.info("--- Application Starting Up ---")
    
    # 1. Create database tables if they don't exist
    logger.info("Verifying database tables...")
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables verified.")

    # 2. Ensure shared directories are present
    ensure_directories_exist()
    logger.info(f"Verified shared directories exist at '{settings.SHARED_AUDIO_PATH}'.")

    # 3. Start background tasks
    redis_task = asyncio.create_task(redis_listener())
    cleanup_task = asyncio.create_task(cleanup_stale_jobs())
    logger.info("Started background tasks for Redis listener and job cleanup.")
    
    yield # The application is now running
    
    # --- Shutdown ---
    logger.info("--- Application Shutting Down ---")
    redis_task.cancel()
    cleanup_task.cancel()
    logger.info("Background tasks cancelled.")


# ===================================================================
#   FastAPI Application Initialization
# ===================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="An integrated API for meeting transcription, diarization, and analysis.",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API routers
app.include_router(meeting.router, prefix=f"{settings.API_V1_STR}/meeting", tags=["Meeting Workflow & Analysis"])
app.include_router(speaker.router, prefix=f"{settings.API_V1_STR}/speaker", tags=["Speaker Enrollment & Management"])

# Add a health check endpoint
@app.get("/health", tags=["Health Check"])
def health_check():
    """A simple endpoint to confirm that the API is running."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}