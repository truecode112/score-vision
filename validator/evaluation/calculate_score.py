from typing import Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import json
from fiber.logging_utils import get_logger
from validator.challenge.challenge_types import GSRResponse, ValidationResult
from validator.db.operations import DatabaseManager
from validator.evaluation.scoring_utils import calculate_processing_time, compute_cv_score
import httpx

logger = get_logger(__name__)

async def calculate_score(
    tasks: List[Dict[str, Any]], 
    client: httpx.AsyncClient,
    validator_hotkey: str
) -> Dict[str, Dict[str, float]]:
    """
    Calculate scores for completed tasks and store them with reward data.
    
    Args:
        tasks: List of completed tasks
        client: httpx.AsyncClient instance
        validator_hotkey: Validator's public key hex
        
    Returns:
        Dict mapping node_id to score components
    """
    try:
        # Initialize database manager
        db_manager = DatabaseManager(Path("validator.db"))
        
        # Track scores for each node
        node_scores = {}  # Using node_id (int) as key
        
        # Process each task
        for task in tasks:
            task_id = task['task_id']
            try:
                node_id = int(task['node_id'])  # Convert to int, handle potential errors
            except (ValueError, TypeError):
                logger.error(f"Invalid node_id in task {task_id}: {task['node_id']}")
                continue
                
            miner_hotkey = task['miner_hotkey']
            task_type = task.get('type')  # This should be 'gsr'
            
            if task_type != 'gsr':
                logger.warning(f"Skipping task {task_id} with unknown type: {task_type}")
                continue
            
            # Get processing time stats for this challenge
            time_stats = db_manager.get_processing_time_stats(str(task_id))
            min_time = time_stats.get('min_time', 1.0)  # Minimum 1 second
            max_time = time_stats.get('max_time', 10.0)  # Maximum 10 seconds
            
            # Get processing time from responses table
            cursor = db_manager.get_connection().cursor()
            try:
                cursor.execute("""
                    SELECT processing_time
                    FROM responses
                    WHERE challenge_id = ? AND miner_hotkey = ?
                """, (str(task_id), miner_hotkey))
                row = cursor.fetchone()
                if row and row[0]:
                    processing_time = float(row[0])
                else:
                    logger.error(f"No processing time found in responses for task {task_id}")
                    continue
            finally:
                cursor.connection.close()
            
            try:
                task_data = json.loads(task['task_returned_data'])
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in task_returned_data for task {task_id}")
                continue
                
            # Calculate score for game state reconstruction
            score = compute_cv_score(task_data, processing_time, {'avg_time': max_time})
            
            # Calculate speed score based on relative position between min and max times
            # If time equals min_time -> score = 1.0
            # If time equals max_time -> score = 0.0
            # Linear interpolation between min and max
            if max_time == min_time:
                speed_score = 1.0  # If all times are the same, give full score
            else:
                # Clamp processing time between min and max
                clamped_time = max(min_time, min(max_time, processing_time))
                # Calculate relative position (1.0 at min_time, 0.0 at max_time)
                speed_score = 1.0 - (clamped_time - min_time) / (max_time - min_time)
            
            # Store response score in database
            try:
                # Create a GSRResponse object with the task data
                response = GSRResponse(
                    challenge_id=task_id,
                    frames=task_data.get('frames', task_data),  # Try to get frames or use entire data
                    processing_time=processing_time
                )
                
                response_id = db_manager.store_response(
                    challenge_id=task_id,
                    miner_hotkey=miner_hotkey,
                    response=response,
                    node_id=node_id,
                    processing_time=processing_time
                )
                
                # Then store the score
                validation_result = ValidationResult(
                    score=score,
                    frame_scores={},  # We don't have frame scores for these tasks
                    feedback=f"Processing time: {processing_time:.2f}s"
                )
                db_manager.store_response_score(
                    response_id=response_id,
                    validation_result=validation_result,
                    validator_hotkey=validator_hotkey
                )
                
            except Exception as e:
                logger.error(f"Error storing score for task {task_id}: {str(e)}")
                continue
                
            logger.info(f"Processed task {task_id} with score {score}")
            
            # Accumulate scores for each node (using int node_id)
            if node_id not in node_scores:
                node_scores[node_id] = {
                    'scores': [],
                    'miner_hotkey': miner_hotkey  # Store the hotkey for reference
                }
            node_scores[node_id]['scores'].append((score, speed_score))
        
        # Prepare final scores with components
        final_scores = {}
        for node_id, node_data in node_scores.items():
            scores = node_data['scores']
            if not scores:
                continue
                
            # Calculate average scores
            quality_scores = [score for score, _ in scores]
            speed_scores = [speed for _, speed in scores]
            
            avg_quality_score = sum(quality_scores) / len(quality_scores)
            avg_speed_score = sum(speed_scores) / len(speed_scores)
            
            # Calculate weighted final score (60% quality, 30% availability, 10% speed)
            final_score = (
                avg_quality_score * 0.6 +  # Quality weight
                1.0 * 0.3 +               # Availability weight (assumed 1.0 for now)
                avg_speed_score * 0.1     # Speed weight
            )
            
            final_scores[str(node_id)] = {  # Convert node_id to string for consistency
                'quality_score': avg_quality_score,
                'speed_scoring_factor': avg_speed_score,
                'availability_factor': 1.0,  # Can be adjusted based on uptime
                'response_time': sum(processing_time for _, _ in scores) / len(scores),
                'final_score': final_score,
                'miner_hotkey': node_data['miner_hotkey']  # Include hotkey in output
            }
            
        return final_scores
        
    except Exception as e:
        logger.error(f"Error in calculate_score: {str(e)}", exc_info=True)
        return {}