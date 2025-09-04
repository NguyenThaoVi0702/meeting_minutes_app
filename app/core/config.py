import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Determine the base directory of the project
# This allows us to safely build paths from the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Manages all application configuration settings.
    Loads settings from a .env file and environment variables.
    """
    # --------------------------------------------------------------------------
    # Core Application Settings
    # --------------------------------------------------------------------------
    PROJECT_NAME: str = "VietinBank AI Meeting Assistant"
    API_V1_STR: str = "/api/v1"
    
    # Load settings from a .env file located at the project root
    model_config = SettingsConfigDict(env_file=os.path.join(BASE_DIR, '.env'), extra='ignore')

    # --------------------------------------------------------------------------
    # Database Configuration
    # --------------------------------------------------------------------------
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str

    @property
    def DATABASE_URL(self) -> str:
        """Constructs the full database connection string."""
        return (f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
                f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}")

    # --------------------------------------------------------------------------
    # Redis Configuration (for Celery & WebSockets)
    # --------------------------------------------------------------------------
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        """Constructs the Redis connection URL for Celery."""
        password = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{password}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # --------------------------------------------------------------------------
    # LLM Service Configuration (LiteLLM)
    # --------------------------------------------------------------------------
    LITE_LLM_API_KEY: str
    LITE_LLM_BASE_URL: str
    LITE_LLM_MODEL_NAME: str
    LIMIT_TURN: int = 6 # Max conversation turns for chat history

    # --------------------------------------------------------------------------
    # File Storage Paths
    # --------------------------------------------------------------------------
    # This path should point to the shared volume defined in docker-compose.yml
    SHARED_AUDIO_PATH: str = "/app/shared_audio"
    
    # Path for storing speaker enrollment voice samples
    ENROLLMENT_SAMPLES_PATH: str = os.path.join(SHARED_AUDIO_PATH, "enrollment_samples")

    # --------------------------------------------------------------------------
    # AI/ML Model Paths
    # --------------------------------------------------------------------------
    # Path to the Faster-Whisper model directory inside the container
    FASTER_WHISPER_MODEL_PATH: str = "/app/models/merged_model_ct2_dir"
    
    # Path to the NeMo Rimecaster model file for speaker embeddings
    RIMECASTER_MODEL_PATH: str = "/app/models/rimecaster.nemo"

    # --------------------------------------------------------------------------
    # Vector Database Configuration (Qdrant)
    # --------------------------------------------------------------------------
    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "meeting_speakers_v2"
    
    # --------------------------------------------------------------------------
    # Diarization Algorithm Parameters
    # --------------------------------------------------------------------------
    DIAR_SEG_DURATION: float = 3.0
    DIAR_SEG_OVERLAP: float = 1.0
    DIAR_KNOWN_THRESH: float = 0.5  # Similarity threshold for identifying known speakers
    DIAR_HAC_THRESH: float = 0.45   # Distance threshold for clustering unknown speakers
    DIAR_MERGE_PAUSE: float = 0.7   # Max pause in seconds to merge segments from same speaker
    DIAR_VAD_THRESH: float = 0.3    # Voice Activity Detection threshold
    ENABLE_VAD=False 

# Instantiate the settings object. This will be imported by other modules.
settings = Settings()

# --- Utility function to create necessary directories on startup ---
def ensure_directories_exist():
    """
    Creates the shared audio and enrollment sample directories if they don't exist.
    This is called from the main application startup event.
    """
    Path(settings.SHARED_AUDIO_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.ENROLLMENT_SAMPLES_PATH).mkdir(parents=True, exist_ok=True)
