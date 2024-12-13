import os
from dotenv import load_dotenv
import uvicorn

from fiber.logging_utils import get_logger
from miner.server import factory_app
from miner.endpoints.soccer import factory_router as soccer_router

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Create the FastAPI app
app = factory_app(debug=True)

# Include the soccer router
app.include_router(soccer_router(), prefix="/soccer")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
