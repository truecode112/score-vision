import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

import httpx
from fiber.logging_utils import get_logger
from validator.db.operations import DatabaseManager
from validator.evaluation.evaluation import GSRValidator
from validator.challenge.challenge_types import GSRResponse, ValidationResult, GSRChallenge, ChallengeType
from validator.evaluation.calculate_score import calculate_score
from validator.utils.api import update_task_scores
from validator.config import VALIDATION_DELAY
from loguru import logger

logger = get_logger(__name__)

async def _evaluate_single_response(
    validator: GSRValidator,
    db_manager: DatabaseManager,
    video_path: Path,
    row: sqlite3.Row
) -> Dict[str, Any]:
    """Evaluate a single response."""
    try:
        # Log row information (excluding response_data)
        row_dict = dict(row)
        if 'response_data' in row_dict:
            row_dict['response_data'] = '<omitted>'
        logger.info(f"Processing row: {row_dict}")
        
        # Get response data
        response_data = json.loads(row["response_data"] if row["response_data"] else "{}")
        logger.info(f"Response metadata - challenge_id: {row['challenge_id']}, miner_hotkey: {row['miner_hotkey']}")
        
        # Create GSRResponse object with node_id
        response = GSRResponse(
            challenge_id=row["challenge_id"],
            miner_hotkey=row["miner_hotkey"],
            frames=response_data.get("frames", {}),
            processing_time=row["processing_time"],
            response_id=row["response_id"],
            node_id=row["node_id"]
        )
        logger.info(f"Created GSRResponse object for response_id: {response.response_id}")

        # Get challenge data
        challenge = GSRChallenge(
            challenge_id=row["challenge_id"],
            type=ChallengeType.GSR,
            created_at=row["created_at"] if "created_at" in row.keys() else None,
            video_url=row["video_url"] if "video_url" in row.keys() else ""
        )
        logger.info(f"Processing challenge_id: {challenge.challenge_id}")

        # Get timing information from database
        started_at = db_manager.get_challenge_assignment_sent_at(challenge.challenge_id, response.miner_hotkey)
        completed_at = row["completed_at"] if "completed_at" in row.keys() else None
        logger.info(f"Timing info - started_at: {started_at}, completed_at: {completed_at}")

        # Evaluate response
        result = await validator.evaluate_response(response, challenge, video_path)
        logger.info(f"Evaluation complete - score: {result.score}")
        
        # Mark response as evaluated
        db_manager.mark_response_as_evaluated(response.response_id)

        return {
            "challenge_id": challenge.challenge_id,
            "miner_hotkey": response.miner_hotkey,
            "node_id": response.node_id,
            "response_id": response.response_id,
            "score": result.score,
            "processing_time": row["processing_time"],
            "validation_result": result,
            "task_returned_data": response_data,
            "started_at": started_at,
            "completed_at": completed_at,
            "received_at": row["received_at"] if "received_at" in row.keys() else None
        }
    except Exception as e:
        logger.error(f"Error evaluating response: {str(e)}")
        logger.error(f"Row object type: {type(row)}")
        if hasattr(row, 'keys'):
            logger.error(f"Available row keys: {row.keys()}")
        raise

async def evaluate_pending_responses(
    db_manager: DatabaseManager,
    validator: GSRValidator,
    batch_size: int = 10,
    sleep_interval: int = 60
) -> None:
    """
    Continuously fetch and evaluate pending responses from the DB.
    Each challenge's responses are evaluated in parallel to speed it up.
    """
    while True:
        try:
            # Pull unevaluated responses grouped by challenge_id
            conn = db_manager.get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                # First get distinct challenge IDs with pending responses
                cursor.execute("""
                    SELECT DISTINCT c.challenge_id, c.video_url, c.type AS challenge_type
                    FROM responses r
                    JOIN challenges c ON r.challenge_id = c.challenge_id
                    WHERE r.evaluated = FALSE
                      AND datetime(r.received_at) <= datetime('now', '-' || ? || ' minutes')
                    LIMIT 1
                """, (VALIDATION_DELAY.total_seconds() / 60,))
                
                challenge = cursor.fetchone()
                if not challenge:
                    logger.debug("No pending challenges to evaluate")
                    await asyncio.sleep(sleep_interval)
                    continue

                # Get all pending responses for this challenge
                cursor.execute("""
                    SELECT 
                        r.response_id,
                        r.challenge_id,
                        r.miner_hotkey,
                        r.node_id,
                        r.response_data,
                        r.processing_time,
                        r.received_at,
                        r.completed_at,
                        c.created_at,
                        c.video_url,
                        c.type AS challenge_type
                    FROM responses r
                    JOIN challenges c ON r.challenge_id = c.challenge_id
                    WHERE r.challenge_id = ?
                      AND r.evaluated = FALSE
                      AND datetime(r.received_at) <= datetime('now', '-' || ? || ' minutes')
                """, (challenge["challenge_id"], VALIDATION_DELAY.total_seconds() / 60))
                
                pending_responses = cursor.fetchall()
                
            finally:
                conn.close()

            if not pending_responses:
                logger.debug(f"No pending responses for challenge {challenge['challenge_id']}")
                await asyncio.sleep(sleep_interval)
                continue
            
            logger.info(f"Processing challenge {challenge['challenge_id']} with {len(pending_responses)} responses")
            
            try:
                # Download video once for the challenge
                video_url = challenge["video_url"]
                logger.info(f"Downloading video for challenge {challenge['challenge_id']}")
                video_path = await validator.download_video(video_url)
                
                challenge_obj = GSRChallenge(
                    challenge_id=challenge["challenge_id"],
                    type=challenge["challenge_type"],
                    video_url=video_url,
                    created_at=datetime.now(timezone.utc)
                )
                
                # Create tasks for parallel evaluation
                tasks = []
                for row in pending_responses:
                    task = _evaluate_single_response(validator, db_manager, video_path, row)
                    tasks.append(task)
                
                logger.info(f"Starting parallel evaluation of {len(tasks)} responses")
                evaluation_results = await asyncio.gather(*tasks, return_exceptions=False)
                logger.info(f"Completed parallel evaluation of responses")
                
                # Calculate scores
                async with httpx.AsyncClient() as client:
                    scores = await calculate_score(evaluation_results, client, validator_hotkey=validator.validator_hotkey, db_manager=db_manager)
                    
                    # Update DB and external API
                    for node_id, score_data in scores.items():
                        response_id = score_data['response_id']
                        miner_hotkey = score_data['miner_hotkey']
                        
                        # Log detailed scoring information
                        logger.info(f"Response {response_id} scoring details:")
                        logger.info(f"  - Quality score: {score_data['quality_score']:.3f}")
                        logger.info(f"  - Speed score: {score_data['speed_score']:.3f}")
                        logger.info(f"  - Availability score: {score_data['availability_score']:.3f}")
                        logger.info(f"  - Final score: {score_data['final_score']:.3f}")
                        
                        # Update response with score and evaluation status
                        db_manager.update_response(
                            response_id=response_id,
                            score=score_data['final_score'],
                            evaluated=True,
                            evaluated_at=datetime.utcnow()
                        )
                        
                        db_manager.store_response_score(
                            response_id=response_id,
                            challenge_id=challenge["challenge_id"],
                            validation_result=score_data['validation_result'],
                            validator_hotkey=validator.validator_hotkey,
                            miner_hotkey=miner_hotkey,
                            node_id=int(node_id),
                            availability_score=score_data['availability_score'],
                            speed_score=score_data['speed_score'],
                            total_score=score_data['final_score']
                        )
                        
                        # Update external API
                        update_success = await update_task_scores(
                            validator_address=validator.validator_hotkey,
                            task_id=challenge["challenge_id"],
                            challenge_id=challenge["challenge_id"],
                            miner_id=node_id,
                            miner_hotkey=score_data['miner_hotkey'],
                            response_data=json.dumps(score_data['task_returned_data']),
                            evaluation_score=score_data['quality_score'],
                            speed_score=score_data['speed_score'],
                            availability_score=score_data['availability_score'],
                            total_score=score_data['final_score'],
                            processing_time=score_data['processing_time'],
                            started_at=score_data['started_at'],
                            completed_at=score_data['completed_at']
                        )
                        
                        if update_success:
                            logger.info(f"Updated API with scores for response {response_id}")
                        else:
                            logger.warning(f"Failed to update API with scores for response {response_id}")
            
            except Exception as e:
                logger.error(f"Error processing challenge {challenge['challenge_id']}: {str(e)}")
                logger.exception("Full error traceback:")
            
            # Short delay before next batch
            await asyncio.sleep(sleep_interval)
        
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
    """Entrypoint that sets up the DB, validator, and runs the loop."""
    db_manager = DatabaseManager(db_path)
    validator = GSRValidator(openai_api_key=openai_api_key, validator_hotkey=validator_hotkey)
    validator.db_manager = db_manager  # Let the validator store frame-level evaluations
    
    try:
        await evaluate_pending_responses(
            db_manager=db_manager,
            validator=validator,
            batch_size=batch_size,
            sleep_interval=sleep_interval
        )
    finally:
        db_manager.close()