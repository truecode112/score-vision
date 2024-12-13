import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ..core.config.config_handler import ConfigHandler
from ..core.models.config_models import Config
from ..core.constants import ENV_CONFIG_PATH

@lru_cache()
def get_validator_config() -> Config:
    """Get validator configuration."""
    config_path = os.getenv(ENV_CONFIG_PATH)
    if not config_path:
        config_path = str(Path(__file__).parent / "config" / "validator_config.yml")
    
    return ConfigHandler(config_path).config

@lru_cache()
def get_database_engine():
    """Get SQLAlchemy database engine."""
    config = get_validator_config()
    return create_engine(
        config.database.url,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
        pool_timeout=config.database.timeout
    )

@lru_cache()
def get_session_factory():
    """Get SQLAlchemy session factory."""
    engine = get_database_engine()
    return sessionmaker(bind=engine)

def get_db_session() -> Session:
    """Get a new database session."""
    SessionFactory = get_session_factory()
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()

def get_challenge_dir() -> Path:
    """Get path to challenge directory."""
    config = get_validator_config()
    challenge_dir = Path(__file__).parent / config.storage.challenge_dir
    challenge_dir.mkdir(parents=True, exist_ok=True)
    return challenge_dir

def get_results_dir() -> Path:
    """Get path to results directory."""
    config = get_validator_config()
    results_dir = Path(__file__).parent / config.storage.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
