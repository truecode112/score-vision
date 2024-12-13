from typing import Optional, Dict, Any
import httpx
from datetime import datetime
import json
from fiber.logging_utils import get_logger
from validator.config import SCORE_VISION_API

logger = get_logger(__name__)

async def get_next_challenge(validator_address: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the next challenge from the API.
    
    Args:
        validator_address: The validator's ss58 address
    
    Returns:
        Dict with video_url and task_id, or None if no challenge available
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SCORE_VISION_API}/api/tasks/next?validator_hotkey={validator_address}")
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Got challenge from API: {data}")
            
            # Only return the fields we need
            return {
                'video_url': data['video_url'],
                'task_id': data['id'],
                'type': 'gsr'  # Hardcode the type since we only support GSR for now
            }
            
    except Exception as e:
        logger.error(f"Error fetching challenge: {str(e)}")
        if isinstance(e, httpx.HTTPError):
            logger.error(f"HTTP Error response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return None

async def update_task_scores(
    validator_address: str,
    task_id: int,
    challenge_id: int,
    miner_id: int,
    miner_hotkey: str,
    response_data: dict,
    evaluation_score: float,
    speed_score: float,
    availability_score: float,
    total_score: float,
    processing_time: float,
    started_at: datetime,
    completed_at: datetime
) -> bool:
    """
    Update task scores in the API.
    """
    try:
        payload = {
            "id": task_id,
            "challenge_id": challenge_id,
            "miner_id": miner_id,
            "miner_hotkey": miner_hotkey,
            "response_data": response_data,
            "evaluation_score": evaluation_score,
            "speed_score": speed_score,
            "availability_score": availability_score,
            "total_score": total_score,
            "processing_time": processing_time,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        url = f"{SCORE_VISION_API}/api/tasks/update"
        params = {"validator_hotkey": validator_address}
        
        logger.info(f"Sending score update to API:")
        logger.info(f"URL: {url}")
        logger.info(f"Params: {params}")
        logger.info(f"Payload:")
        logger.info(f"  - task_id: {task_id}")
        logger.info(f"  - challenge_id: {challenge_id}")
        logger.info(f"  - miner_id: {miner_id}")
        logger.info(f"  - miner_hotkey: {miner_hotkey}")
        logger.info(f"  - evaluation_score: {evaluation_score}")
        logger.info(f"  - speed_score: {speed_score}")
        logger.info(f"  - availability_score: {availability_score}")
        logger.info(f"  - total_score: {total_score}")
        logger.info(f"  - processing_time: {processing_time}")
        logger.info(f"  - started_at: {started_at.isoformat()}")
        logger.info(f"  - completed_at: {completed_at.isoformat()}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                params=params,
                json=payload
            )
            response.raise_for_status()
            
            logger.info(f"API Response status: {response.status_code}")
            #logger.info(f"API Response body: {response.text}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error updating task scores: {str(e)}")
        if isinstance(e, httpx.HTTPError):
            logger.error(f"HTTP Error response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return False
