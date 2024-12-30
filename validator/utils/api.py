from typing import Optional, Dict, Any
import httpx
from datetime import datetime
import json
from fiber.logging_utils import get_logger
from validator.config import SCORE_VISION_API
import asyncio

logger = get_logger(__name__)

def optimize_bbox_coordinates(bbox):
    """Round bbox coordinates to 2 decimal places to reduce payload size."""
    return [round(float(x), 2) for x in bbox]

def optimize_keypoints(keypoints):
    """Round keypoint coordinates to 2 decimal places."""
    return [round(float(x), 2) for x in keypoints]

def optimize_response_data(response_data: dict) -> dict:
    """
    Optimize response data to reduce payload size.
    - Rounds coordinates to 2 decimal places
    - Removes unnecessary metadata
    - Optimizes data structure
    """
    optimized_data = {}
    
    for frame_id, frame_data in response_data.get("frames", {}).items():
        optimized_frame = {}
        
        # Optimize players data
        if "players" in frame_data:
            optimized_frame["players"] = []
            for player in frame_data["players"]:
                optimized_player = {
                    "bbox": optimize_bbox_coordinates(player["bbox"]),
                    "class_id": player.get("class_id", 2)  # Default to regular player
                }
                optimized_frame["players"].append(optimized_player)
        
        # Optimize ball data
        if "ball" in frame_data:
            optimized_frame["ball"] = []
            for ball in frame_data["ball"]:
                optimized_ball = {
                    "bbox": optimize_bbox_coordinates(ball["bbox"])
                }
                optimized_frame["ball"].append(optimized_ball)
        
        # Optimize keypoints
        if "keypoints" in frame_data:
            optimized_frame["keypoints"] = [
                optimize_keypoints(point) for point in frame_data["keypoints"]
            ]
        
        optimized_data[frame_id] = optimized_frame
    
    return {
        "frames": optimized_data,
        "challenge_id": response_data.get("challenge_id"),
        "processing_time": round(float(response_data.get("processing_time", 0)), 2)
    }

def log_data_size(data: Dict, prefix: str = "") -> None:
    """Log the size of data and its components."""
    try:
        # Convert to JSON string to get actual payload size
        data_json = json.dumps(data)
        total_size = len(data_json)
        
        logger.info(f"{prefix}Total payload size: {total_size / 1024:.2f} KB ({total_size:,} bytes)")
        
        # Log sizes of main components
        if isinstance(data, dict):
            for key, value in data.items():
                component_size = len(json.dumps(value))
                if component_size > 1024:  # Only log components larger than 1KB
                    logger.info(f"{prefix}{key}: {component_size / 1024:.2f} KB")
                    
                    # For frames data, provide more detailed breakdown
                    if key == "frames" and isinstance(value, dict):
                        total_keypoints = 0
                        total_players = 0
                        total_balls = 0
                        
                        for frame_data in value.values():
                            if isinstance(frame_data, dict):
                                keypoints = frame_data.get("keypoints", [])
                                players = frame_data.get("players", [])
                                balls = frame_data.get("ball", [])
                                
                                total_keypoints += len(keypoints)
                                total_players += len(players)
                                total_balls += len(balls)
                        
                        logger.info(f"{prefix}Frame stats:")
                        logger.info(f"{prefix}- Total frames: {len(value)}")
                        logger.info(f"{prefix}- Total keypoints: {total_keypoints}")
                        logger.info(f"{prefix}- Total players detected: {total_players}")
                        logger.info(f"{prefix}- Total balls detected: {total_balls}")
        
        # Warn if approaching size limit
        if total_size > 900000:  # 900KB warning threshold
            logger.warning(f"Payload size ({total_size / 1024:.2f} KB) is approaching the 1MB limit!")
            
    except Exception as e:
        logger.error(f"Error logging data size: {str(e)}")

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
    Update task scores in the API with retry mechanism.
    """
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # Log original data size
            logger.info("Original response data size:")
            log_data_size(response_data, prefix="Original: ")
            
            # Optimize response data to reduce payload size
            optimized_response = optimize_response_data(response_data)
            
            # Log optimized data size
            logger.info("Optimized response data size:")
            log_data_size(optimized_response, prefix="Optimized: ")
            
            # Calculate size reduction
            original_size = len(json.dumps(response_data))
            optimized_size = len(json.dumps(optimized_response))
            reduction_percent = ((original_size - optimized_size) / original_size) * 100
            logger.info(f"Size reduction: {reduction_percent:.1f}%")
            
            payload = {
                "id": task_id,
                "challenge_id": challenge_id,
                "miner_id": miner_id,
                "miner_hotkey": miner_hotkey,
                "response_data": optimized_response,
                "evaluation_score": round(evaluation_score, 4),
                "speed_score": round(speed_score, 4),
                "availability_score": round(availability_score, 4),
                "total_score": round(total_score, 4),
                "processing_time": round(processing_time, 2),
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "created_at": datetime.now().isoformat()
            }
            
            # Log final payload size
            logger.info("Final API payload size:")
            log_data_size(payload, prefix="Final: ")
            
            url = f"{SCORE_VISION_API}/api/tasks/update"
            params = {"validator_hotkey": validator_address}
            
            logger.info(f"Sending score update to API (Attempt {attempt + 1}/{max_retries}):")
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
            return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"Task assignment not found for task_id: {task_id}. Skipping update.")
                return False
            logger.error(f"HTTP error occurred (Attempt {attempt + 1}/{max_retries}): {str(e)}")
        except Exception as e:
            logger.error(f"Error updating task scores (Attempt {attempt + 1}/{max_retries}): {str(e)}")

        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)

    logger.error(f"Failed to update task scores after {max_retries} attempts. Skipping update.")
    return False
