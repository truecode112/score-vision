import os
import sys
import threading
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from fiber.logging_utils import get_logger
from fiber.miner.middleware import configure_extra_logging_middleware
from miner.core import configuration
from miner.endpoints.soccer import factory_router as soccer_router
from miner.endpoints.availability import router as availability_router

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

miner_lock = asyncio.Lock()

def create_app(debug: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.miner_lock = miner_lock
        
        config = configuration.factory_config()
        metagraph = config.metagraph
        sync_thread = None
        if metagraph.substrate is not None:
            sync_thread = threading.Thread(target=metagraph.periodically_sync_nodes, daemon=True)
            sync_thread.start()

        yield

        logger.info("Shutting down...")
        metagraph.shutdown()
        if metagraph.substrate is not None and sync_thread is not None:
            sync_thread.join()

    app = FastAPI(lifespan=lifespan, debug=debug)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add development middleware if needed
    if debug:
        configure_extra_logging_middleware(app)

    # Include the soccer router
    app.include_router(soccer_router(), prefix="/soccer")
    
    # Include the availability router at root level
    app.include_router(availability_router)

    return app

# Create the FastAPI app
app = create_app(debug=True)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6999) 