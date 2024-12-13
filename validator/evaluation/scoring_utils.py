from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import json
from fiber.logging_utils import get_logger
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from pytz import UTC

logger = get_logger(__name__)

# Constants for scoring
W_PREDICTION = 1.0
W_COMPUTER_VISION = 2.0

def validate_prediction_timing(prediction_time: datetime, kickoff_time: datetime) -> bool:
    """
    Validate that the prediction was made before the match kickoff.
    
    Args:
        prediction_time: When the prediction was received
        kickoff_time: When the match started
        
    Returns:
        bool: True if prediction was made before kickoff, False otherwise
    """
    try:
        # Add a small buffer (e.g., 1 minute) to account for network delays
        SUBMISSION_BUFFER = timedelta(minutes=1)
        
        # Convert kickoff_time to datetime if it's a string
        if isinstance(kickoff_time, str):
            kickoff_time = datetime.fromisoformat(kickoff_time.replace('Z', '+00:00'))
            
        # Ensure both times are timezone aware for comparison
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=UTC)
        if kickoff_time.tzinfo is None:
            kickoff_time = kickoff_time.replace(tzinfo=UTC)
            
        # Debug logging
        #logger.debug(f"Comparing prediction time {prediction_time} with kickoff time {kickoff_time}")
        
        return prediction_time + SUBMISSION_BUFFER <= kickoff_time
        
    except Exception as e:
        logger.error(f"Error validating prediction timing: {str(e)}")
        return False

def compute_prediction_score(
    prediction_data: Dict[str, Any], 
    processing_time: float, 
    time_stats: Dict[str, float],
    prediction_accuracy: float = 0.0
) -> float:
    """
    Compute score for a prediction task.
    
    Args:
        prediction_data: Dictionary containing prediction details
        processing_time: Actual processing time in seconds
        time_stats: Dictionary containing min/max/avg processing times
        prediction_accuracy: Accuracy of the prediction (0.0 for incorrect, 1.0 for correct)
        
    Returns:
        float: Computed score between 0 and 1
    """
    try:
        # Extract confidence (between 0 and 1)
        confidence = float(prediction_data.get('confidence', 0.5))
        
        if prediction_accuracy == 1.0:
            # Correct prediction: higher confidence gives higher score
            # Score range: 0.5 to 1.0 based on confidence
            base_score = 0.5 + (0.5 * confidence)
        else:
            # Incorrect prediction: higher confidence gives lower score
            # Score range: 0.0 to 0.1 based on inverse of confidence
            base_score = 0.1 * (1 - confidence)
        
        # Apply prediction weight
        return base_score * W_PREDICTION
        
    except Exception as e:
        logger.error(f"Error computing prediction score: {str(e)}")
        return 0.0

def compute_cv_score(cv_data: Dict[str, Any], processing_time: float, time_stats: Dict[str, float]) -> float:
    """
    Compute score for a computer vision task.
    
    Args:
        cv_data: Dictionary containing CV task results
        processing_time: Actual processing time in seconds
        time_stats: Dictionary containing min/max/avg processing times
        
    Returns:
        float: Computed score between 0 and 1
    """
    try:
        # Check if required outputs exist
        has_video = bool(cv_data.get('processed_video_url'))
        has_data = bool(cv_data.get('positional_data'))
        
        # Calculate time factor using min/max times from past 24h
        min_time = time_stats.get('min_time', processing_time)
        max_time = time_stats.get('max_time', processing_time)
        
        # Normalize processing time between 0 and 1
        if max_time == min_time:
            time_factor = 1.0
        else:
            time_factor = 1.0 - ((processing_time - min_time) / (max_time - min_time))
            time_factor = max(0.0, min(1.0, time_factor))  # Clamp between 0 and 1
        
        # Compute completeness score
        completeness = (has_video + has_data) / 2
        
        # Combine factors with higher weight on completeness for CV tasks
        base_score = 0.7 * completeness + 0.3 * time_factor
        
        # Apply CV weight
        return base_score * W_COMPUTER_VISION
        
    except Exception as e:
        logger.error(f"Error computing CV score: {str(e)}")
        return 0.0

def calculate_processing_time(sent_at: str, received_at: str) -> Optional[float]:
    """
    Calculate processing time in seconds between sent_at and received_at.
    
    Args:
        sent_at: ISO format timestamp string
        received_at: ISO format timestamp string
        
    Returns:
        float: Processing time in seconds, or None if calculation fails
    """
    try:
        start_time = datetime.fromisoformat(sent_at.rstrip('Z'))
        end_time = datetime.fromisoformat(received_at.rstrip('Z'))
        return (end_time - start_time).total_seconds()
    except Exception as e:
        logger.error(f"Error calculating processing time: {str(e)}")
        return None

def aggregate_scores(scores: List[Tuple[int, float, str]], window_hours: int = 24) -> Dict[int, float]:
    """
    Aggregate scores for each miner over a time window.
    
    Args:
        scores: List of tuples (node_id, score, task_type)
        window_hours: Number of hours to look back
        
    Returns:
        Dict mapping node_id to aggregated score
    """
    aggregated = {}
    
    for node_id, score, task_type in scores:
        if node_id not in aggregated:
            aggregated[node_id] = 0.0
            
        # Add weighted score
        if task_type == 'predict_match':
            aggregated[node_id] += score * W_PREDICTION
        elif task_type == 'voronois':
            aggregated[node_id] += score * W_COMPUTER_VISION
            
    return aggregated
