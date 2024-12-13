from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.dependencies import get_config
from miner.endpoints.soccer import miner_lock

logger = get_logger(__name__)

router = APIRouter()

@router.get("/availability")
async def check_availability(
    request: Request,
    validator_ss58_address: str = Header(..., alias="validator-hotkey"),
    config: Config = Depends(get_config)
):
    """Check if miner is available to process soccer challenges"""
    try:
        is_available = not miner_lock.locked()
        logger.debug(f"Soccer miner availability check from validator {validator_ss58_address}: {is_available}")
        return {"available": is_available}
    except Exception as e:
        logger.error(f"Error checking availability: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 