import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from dev.env
validator_dir = Path(__file__).parent
dev_env_path = validator_dir / "dev.env"
load_dotenv(dev_env_path)

# Network configuration
NETUID = int(os.getenv("NETUID", "1"))
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "test")
SUBTENSOR_ADDRESS = os.getenv("SUBTENSOR_ADDRESS", "127.0.0.1:9944")

# Validator configuration
HOTKEY_NAME = os.getenv("HOTKEY_NAME", "default")
WALLET_NAME = os.getenv("WALLET_NAME", "default")
MIN_STAKE_THRESHOLD = float(os.getenv("MIN_STAKE_THRESHOLD", "2"))
ENV = os.getenv("ENV", "dev")
IS_VALIDATOR = os.getenv("IS_VALIDATOR", "True").lower() == "true"
VALIDATOR_PORT = int(os.getenv("VALIDATOR_PORT", "8000"))
VALIDATOR_HOST = os.getenv("VALIDATOR_HOST", "0.0.0.0")
MIN_MINERS = int(os.getenv("MIN_MINERS", "2"))
MAX_MINERS = int(os.getenv("MAX_MINERS", "100"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.7"))
FRAMES_TO_VALIDATE = int(os.getenv("FRAMES_TO_VALIDATE", "2"))

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API Configuration
SCORE_VISION_API = os.getenv("SCORE_VISION_API", "http://localhost:8000")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# Weights
WEIGHT_EVALUATION = float(os.getenv("WEIGHT_EVALUATION", "0.6"))
WEIGHT_AVAILABILITY = float(os.getenv("WEIGHT_AVAILABILITY", "0.3"))
WEIGHT_SPEED = float(os.getenv("WEIGHT_SPEED", "0.1"))

# Additional settings needed for operation
CHALLENGE_INTERVAL = timedelta(minutes=int(os.getenv("CHALLENGE_INTERVAL_MINUTES", "5")))
CHALLENGE_TIMEOUT = timedelta(minutes=int(os.getenv("CHALLENGE_TIMEOUT_MINUTES", "360")))
DB_PATH = Path(os.getenv("DB_PATH", "validator.db"))
WEIGHTS_INTERVAL = timedelta(minutes=int(os.getenv("WEIGHTS_INTERVAL_MINUTES", "21")))
VALIDATION_DELAY = timedelta(minutes=int(os.getenv("VALIDATION_DELAY_MINUTES", "10")))

# Log initial configuration
import logging
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

logger.info("Validator Configuration:")
logger.info(f"Network: {SUBTENSOR_NETWORK}")
logger.info(f"Netuid: {NETUID}")
logger.info(f"Min miners: {MIN_MINERS}")
logger.info(f"Max miners: {MAX_MINERS}")
logger.info(f"Min stake threshold: {MIN_STAKE_THRESHOLD}")
logger.info(f"Score threshold: {SCORE_THRESHOLD}")
logger.info(f"Frames to validate: {FRAMES_TO_VALIDATE}")
logger.info(f"Challenge interval: {CHALLENGE_INTERVAL}")
logger.info(f"Challenge timeout: {CHALLENGE_TIMEOUT}")
logger.info(f"Weights interval: {WEIGHTS_INTERVAL}")
logger.info(f"DB path: {DB_PATH}")
logger.info(f"Environment: {ENV}")
logger.info(f"Log level: {LOG_LEVEL}")
logger.info(f"Weight evaluation: {WEIGHT_EVALUATION}")
logger.info(f"Weight availability: {WEIGHT_AVAILABILITY}")
logger.info(f"Weight speed: {WEIGHT_SPEED}")
logger.info(f"Validation delay: {VALIDATION_DELAY}")
