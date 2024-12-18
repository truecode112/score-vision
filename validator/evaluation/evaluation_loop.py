import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List
from pathlib import Path

from fiber.logging_utils import get_logger
from validator.db.operations import DatabaseManager
from validator.evaluation.evaluation import GSRValidator
from validator.challenge.challenge_types import GSRResponse, ValidationResult, GSRChallenge
from validator.evaluation.calculate_score import calculate_score
from validator.utils.api import update_task_scores
import httpx
from validator.config import VALIDATION_DELAY

logger = get_logger(__name__)

async def evaluate_pending_responses(
    db_manager: DatabaseManager,
    validator: GSRValidator,
    batch_size: int = 10,
    sleep_interval: int = 60  # 1 minute
) -> None:
    """Continuously evaluate pending responses in the database."""
    
    while True:
        try:
            # Get unevaluated responses
            conn = db_manager.get_connection()
            conn.row_factory = sqlite3.Row  # Enable dictionary access
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT 
                        r.response_id,
                        r.challenge_id,
                        r.miner_hotkey,
                        r.node_id,
                        r.response_data,
                        r.processing_time,
                        c.video_url,
                        c.type as challenge_type
                    FROM responses r
                    JOIN challenges c ON r.challenge_id = c.challenge_id
                    WHERE r.evaluated = FALSE
                    AND datetime(r.received_at) <= datetime('now', '-' || ? || ' minutes')
                    LIMIT ?
                """, (VALIDATION_DELAY.total_seconds() / 60, batch_size))
                
                pending_responses = cursor.fetchall()
                
            finally:
                conn.close()
            
            if not pending_responses:
                logger.debug("No pending responses to evaluate")
                await asyncio.sleep(sleep_interval)
                continue
                
            logger.info(f"Found {len(pending_responses)} pending responses to evaluate")
            
            # Process each response
            for row in pending_responses:
                response_id = row['response_id']
                challenge_id = row['challenge_id']
                miner_hotkey = row['miner_hotkey']
                node_id = row['node_id']
                response_data = row['response_data']
                processing_time = row['processing_time']
                video_url = row['video_url']
                challenge_type = row['challenge_type']
                
                try:
                    # Download video for evaluation
                    logger.info(f"Downloading video for response {response_id}")
                    video_path = await validator.download_video(video_url)
                    
                    # Parse response data
                    if isinstance(response_data, str):
                        response_data = json.loads(response_data)
                    
                    logger.debug(f"Raw response data structure: {type(response_data)}")
                    if isinstance(response_data, dict):
                        logger.debug(f"Response data keys: {response_data.keys()}")
                    
                    # Ensure frames data is a dictionary
                    frames_data = response_data.get('frames', {})
                    logger.debug(f"Frames data structure: {type(frames_data)}")
                    if not isinstance(frames_data, dict):
                        frames_data = {}
                        logger.warning(f"Response frames data was not a dictionary (was {type(frames_data)}), defaulting to empty dict")
                    
                    # Create GSRResponse object
                    response = GSRResponse(
                        challenge_id=challenge_id,
                        frames=frames_data,
                        processing_time=response_data.get('processing_time', 0.0),
                        node_id=node_id,
                        miner_hotkey=miner_hotkey,
                        response_id=response_id,
                        received_at=datetime.now(timezone.utc)
                    )
                    
                    # Create challenge object
                    challenge = GSRChallenge(
                        challenge_id=challenge_id,
                        type=challenge_type,
                        video_url=video_url,
                        created_at=datetime.now(timezone.utc)
                    )
                    
                    # Get sent_at from challenge_assignments
                    conn = db_manager.get_connection()
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            SELECT sent_at
                            FROM challenge_assignments
                            WHERE challenge_id = ? AND miner_hotkey = ?
                            AND status = 'sent'
                        """, (str(challenge_id), miner_hotkey))
                        assignment_row = cursor.fetchone()
                        sent_at = datetime.fromisoformat(assignment_row['sent_at']) if assignment_row and assignment_row['sent_at'] else None
                    finally:
                        conn.close()
                    
                    # Use validator's built-in evaluation
                    logger.info(f"Evaluating response {response_id} from miner {miner_hotkey}")
                    validation_result = await validator.evaluate_response(
                        response=response,
                        challenge=challenge,
                        video_path=video_path
                    )
                    
                    # Calculate scores including speed and availability
                    async with httpx.AsyncClient() as client:
                        task_data = [{
                            'task_id': challenge_id,
                            'node_id': node_id,
                            'miner_hotkey': miner_hotkey,
                            'type': 'gsr',
                            'sent_at': sent_at.isoformat() if sent_at else None,
                            'received_at': response.received_at.isoformat() if response.received_at else None,
                            'task_returned_data': json.dumps(response_data)
                        }]
                        
                        scores = await calculate_score(task_data, client, validator_hotkey=validator.validator_hotkey)
                        if scores and str(node_id) in scores:
                            node_score = scores[str(node_id)]
                            
                            # Store the evaluation result with all scores
                            db_manager.store_response_score(
                                response_id=response_id,
                                validation_result=validation_result,
                                validator_hotkey=validator.validator_hotkey,
                                availability_score=node_score['availability_factor'],
                                speed_score=node_score['speed_scoring_factor'],
                                total_score=node_score['final_score']
                            )
                            
                            # Update API with scores
                            await update_task_scores(
                                validator_address=validator.validator_hotkey,
                                task_id=challenge_id,
                                challenge_id=challenge_id,
                                miner_id=node_id,
                                miner_hotkey=miner_hotkey,
                                response_data=response_data,
                                evaluation_score=validation_result.score,
                                speed_score=node_score['speed_scoring_factor'],
                                availability_score=node_score['availability_factor'],
                                total_score=node_score['final_score'],
                                processing_time=node_score['response_time'],
                                started_at=sent_at or datetime.now(timezone.utc),
                                completed_at=response.received_at or datetime.now(timezone.utc)
                            )
                            
                            logger.info(f"Updated API with scores for response {response_id}")
                    
                    logger.info(f"Stored evaluation result for response {response_id}: score={validation_result.score}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating response {response_id}: {str(e)}")
                    logger.exception("Full error traceback:")
                    continue
            
            # Small delay between batches
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in evaluation loop: {str(e)}")
            logger.exception("Full error traceback:")
            await asyncio.sleep(sleep_interval)

async def run_evaluation_loop(
    db_path: str,
    openai_api_key: str,
    validator_hotkey: str,
    batch_size: int = 10,
    sleep_interval: int = 60
) -> None:
    """Run the evaluation loop."""
    db_manager = DatabaseManager(db_path)
    validator = GSRValidator(openai_api_key=openai_api_key, validator_hotkey=validator_hotkey)
    validator.db_manager = db_manager  # Add db_manager to validator
    
    try:
        await evaluate_pending_responses(
            db_manager=db_manager,
            validator=validator,
            batch_size=batch_size,
            sleep_interval=sleep_interval
        )
    finally:
        db_manager.close() 