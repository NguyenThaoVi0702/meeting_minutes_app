"""
This module initializes the database engine.

The `engine` object created here is the central access point to the
database for the entire application. It's configured using the
DATABASE_URL from the application settings.

The import of `app.db.models` is crucial as it ensures that all
SQLModel classes are registered with the metadata before any database
operations (like table creation) are attempted.
"""

from sqlmodel import create_engine
from app.core.config import settings

# Create the database engine.
# The `echo=False` argument disables verbose SQL logging in production.
# For debugging, you can temporarily set this to True.
engine = create_engine(settings.DATABASE_URL, echo=False)


# This line is intentionally placed at the bottom.
# By importing the `models` module, we ensure that all classes inheriting from
# SQLModel are registered with SQLAlchemy's metadata object before the
# application tries to use them (e.g., in `main.py` on startup to create tables).
from app.db import models