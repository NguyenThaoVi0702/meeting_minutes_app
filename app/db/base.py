from sqlmodel import create_engine
from app.core.config import settings
engine = create_engine(settings.DATABASE_URL, echo=True) #Enable verbose logging temporarily
from app.db import models
