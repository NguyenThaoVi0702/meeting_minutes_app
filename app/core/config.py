import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Manages all application configuration settings.
    Loads settings from a .env file and environment variables.
    """

    PROJECT_NAME: str = "VietinBank AI Meeting Assistant"
    API_V1_STR: str = "/api/v1"
    
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
    REDIS_HOST: str
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
    LIMIT_TURN: int = 6 

    # --------------------------------------------------------------------------
    # File Storage Paths
    # --------------------------------------------------------------------------

    SHARED_AUDIO_PATH: str = "/code/shared_audio"
    ENROLLMENT_SAMPLES_PATH: str = os.path.join(SHARED_AUDIO_PATH, "enrollment_samples")

    # --------------------------------------------------------------------------
    # AI/ML Model Paths
    # --------------------------------------------------------------------------
    # Path to the Faster-Whisper model 
    FASTER_WHISPER_MODEL_PATH: str = "/app/models/merged_model_ct2_dir"
    
    # Path to the NeMo Rimecaster model
    RIMECASTER_MODEL_PATH: str = "/app/models/rimecaster.nemo"

    # --------------------------------------------------------------------------
    # Vector Database Configuration (Qdrant)
    # --------------------------------------------------------------------------
    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "meeting_speakers"
    
    # --------------------------------------------------------------------------
    # Diarization Algorithm Parameters
    # --------------------------------------------------------------------------
    DIAR_SEG_DURATION: float = 3.0
    DIAR_SEG_OVERLAP: float = 1.0
    DIAR_KNOWN_THRESH: float = 0.5 
    DIAR_HAC_THRESH: float = 0.45   
    DIAR_MERGE_PAUSE: float = 0.7  
    DIAR_VAD_THRESH: float = 0.3   
    ENABLE_VAD: bool = True 


settings = Settings()


def ensure_directories_exist():
    """
    Creates the shared audio and enrollment sample directories if they don't exist.
    """
    Path(settings.SHARED_AUDIO_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.ENROLLMENT_SAMPLES_PATH).mkdir(parents=True, exist_ok=True)
