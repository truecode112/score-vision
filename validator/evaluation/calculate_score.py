from typing import Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import json
from fiber.logging_utils import get_logger
from validator.challenge.challenge_types import GSRResponse, ValidationResult
from validator.db.operations import DatabaseManager
import httpx

logger = get_logger(__name__)

async def calculate_score(
    evaluation_results: List[Dict[str, Any]], 
    client: httpx.AsyncClient,
    validator_hotkey: str,
    db_manager: DatabaseManager
) -> Dict[str, Dict[str, float]]:
    """
    Calculate scores for completed tasks and store them with reward data.
    
    Args:
        evaluation_results: List of evaluation results for a single challenge
        client: httpx.AsyncClient instance
        validator_hotkey: Validator's public key hex
        db_manager: DatabaseManager instance
        
    Returns:
        Dict mapping node_id to score components
    """
    try:
        # Track scores for each node
        node_scores = {}  # Using node_id (str) as key
        
        # Collect all processing times for the challenge
        task_processing_times = [result['processing_time'] for result in evaluation_results]
        
        # Calculate relative processing times
        min_time = min(task_processing_times)
        max_time = max(task_processing_times)
        
        # Process each evaluation result
        for result in evaluation_results:
            node_id = str(result['node_id'])
            miner_hotkey = result['miner_hotkey']
            processing_time = result['processing_time']  # Use directly from response table
            
            # Get quality score from validation result
            quality_score = result['validation_result'].score
            
            # Calculate speed score
            speed_score = calculate_speed_score(processing_time, min_time, max_time)
            
            # Get availability score
            availability_score = db_manager.get_availability_score(int(node_id))
            
            # Calculate final score
            final_score = (
                quality_score * 0.6 +
                speed_score * 0.2 +
                availability_score * 0.2
            )
            
            # Parse response data to ensure it's a dict
            try:
                # If it's already a dict, use it as is
                if isinstance(result['task_returned_data'], dict):
                    response_data = result['task_returned_data']
                # If it's a string, try to parse it
                elif isinstance(result['task_returned_data'], str):
                    response_data = json.loads(result['task_returned_data'])
                else:
                    response_data = {}
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning(f"Could not parse response data for node {node_id}, using empty dict")
                response_data = {}
            
            # Ensure timestamps are properly formatted
            started_at = result.get('started_at')
            completed_at = result.get('completed_at')
            
            # Convert to ISO format string if needed
            if started_at:
                if isinstance(started_at, datetime):
                    started_at = started_at.isoformat()
                elif isinstance(started_at, str):
                    # Assume it's already in ISO format
                    started_at = started_at
                else:
                    started_at = None
                    
            if completed_at:
                if isinstance(completed_at, datetime):
                    completed_at = completed_at.isoformat()
                elif isinstance(completed_at, str):
                    # Assume it's already in ISO format
                    completed_at = completed_at
                else:
                    completed_at = None
            
            # Store scores for each node
            node_scores[node_id] = {
                'response_id': result['response_id'],
                'miner_hotkey': miner_hotkey,
                'quality_score': quality_score,
                'speed_score': speed_score,
                'availability_score': availability_score,
                'final_score': final_score,
                'processing_time': float(processing_time),  # Ensure it's a float
                'validation_result': result['validation_result'],
                'task_returned_data': response_data,
                'started_at': started_at,  # Will be None or ISO format string
                'completed_at': completed_at  # Will be None or ISO format string
            }
        
        return node_scores
        
    except Exception as e:
        logger.error(f"Error in calculate_score: {str(e)}", exc_info=True)
        return {}

def calculate_speed_score(processing_time: float, min_time: float, max_time: float) -> float:
    """Calculate speed score based on processing time."""
    if max_time == min_time:
        return 1.0  # If all times are the same, give full score
    normalized_time = (processing_time - min_time) / (max_time - min_time)
    return 1.0 - normalized_time  # Invert so faster times get higher scores
