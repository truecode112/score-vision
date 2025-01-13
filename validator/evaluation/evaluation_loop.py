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
import cv2

logger = get_logger(__name__)

async def _evaluate_single_response(
    validator: GSRValidator,
    db_manager: DatabaseManager,
    video_path: Path,
    row: sqlite3.Row,
    frame_cache: Dict = None,
    frames_to_validate: List[int] = None
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

        # Evaluate response using cached frames if available
        result = await validator.evaluate_response(
            response, 
            challenge, 
            video_path, 
            frame_cache=frame_cache,
            frames_to_validate=frames_to_validate
        )
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
    sleep_interval: int = 30
) -> None:
    """
    Continuously fetch and evaluate pending responses from the DB.
    Each challenge's responses are evaluated in parallel to speed it up.
    """
    logger.info("Starting evaluation loop")
    while True:
        try:
            
            # Pull unevaluated responses grouped by challenge_id
            conn = db_manager.get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                # Get total number of unevaluated responses
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM responses
                    WHERE evaluated = FALSE
                """)
                total_pending = cursor.fetchone()['count']
                logger.info(f"Total unevaluated responses in DB: {total_pending}")
                
                logger.info("Checking for pending challenges ready for evaluation...")
                # First get distinct challenge IDs with pending responses
                cursor.execute("""
                    SELECT DISTINCT c.challenge_id, c.video_url, c.type AS challenge_type, 
                           COUNT(r.response_id) as pending_count,
                           MIN(r.received_at) as earliest_received
                    FROM responses r
                    JOIN challenges c ON r.challenge_id = c.challenge_id
                    WHERE r.evaluated = FALSE
                      AND datetime(r.received_at) <= datetime('now', '-' || ? || ' minutes')
                    GROUP BY c.challenge_id, c.video_url, c.type
                    LIMIT 1
                """, (VALIDATION_DELAY.total_seconds() / 60,))
                
                challenge = cursor.fetchone()
                if not challenge:
                    logger.info("No challenges ready for evaluation yet")
                    logger.info(f"Sleeping for {sleep_interval} seconds...")
                    await asyncio.sleep(sleep_interval)
                    continue

                logger.info(f"Found challenge {challenge['challenge_id']}:")
                logger.info(f"  - Pending responses: {challenge['pending_count']}")
                logger.info(f"  - Earliest received: {challenge['earliest_received']}")
                logger.info(f"  - Video URL: {challenge['video_url']}")

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
                        c.type AS challenge_type,
                        datetime('now', '-' || ? || ' minutes') as threshold_time
                    FROM responses r
                    JOIN challenges c ON r.challenge_id = c.challenge_id
                    WHERE r.challenge_id = ?
                      AND r.evaluated = FALSE
                      AND datetime(r.received_at) <= datetime('now', '-' || ? || ' minutes')
                """, (VALIDATION_DELAY.total_seconds() / 60, challenge["challenge_id"], VALIDATION_DELAY.total_seconds() / 60))
                
                pending_responses = cursor.fetchall()
                
            finally:
                cursor.close()
                conn.close()

            if not pending_responses:
                logger.info(f"No responses ready for evaluation yet for challenge {challenge['challenge_id']}")
                logger.info(f"Sleeping for {sleep_interval} seconds...")
                await asyncio.sleep(sleep_interval)
                continue
            
            logger.info(f"Processing challenge {challenge['challenge_id']} with {len(pending_responses)} responses")
            
            try:
                # Download video once for the challenge
                video_url = challenge["video_url"]
                logger.info(f"Downloading video for challenge {challenge['challenge_id']}")
                video_path = await validator.download_video(video_url)

                # Generate random frames once for this challenge
                frames_to_validate = validator.select_random_frames(video_path)
                logger.info(f"Selected frames for validation: {frames_to_validate}")

                # Pre-process reference counts for these frames
                frame_cache = {}
                frame_images = []
                for frame_idx in frames_to_validate:
                    cap = cv2.VideoCapture(str(video_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        frame_images.append(frame)
                        frame_cache[frame_idx] = {'frame': frame}

                # Get reference counts in batches for all frames at once
                logger.info("Getting reference counts for all frames...")
                reference_counts = await validator.get_reference_counts_batch(frame_images)
                
                # Update frame cache with reference counts
                for frame_idx, counts in zip(frames_to_validate, reference_counts):
                    frame_cache[frame_idx]['reference_counts'] = counts

                logger.info("Starting parallel evaluation of responses...")
                # Create tasks for parallel evaluation
                tasks = []
                for row in pending_responses:
                    task = _evaluate_single_response(
                        validator=validator,
                        db_manager=db_manager,
                        video_path=video_path,
                        row=row,
                        frame_cache=frame_cache,
                        frames_to_validate=frames_to_validate
                    )
                    tasks.append(task)
                
                evaluation_results = await asyncio.gather(*tasks, return_exceptions=False)
                logger.info(f"Completed parallel evaluation of {len(tasks)} responses")
                
                # Calculate scores and update DB
                async with httpx.AsyncClient() as client:
                    scores = await calculate_score(evaluation_results, client, validator_hotkey=validator.validator_hotkey, db_manager=db_manager)
                    
                    # Log all scores being processed
                    logger.info(f"Processing scores for {len(scores)} responses")
                    
                    # Update DB and external API for each response
                    for response_id, score_data in scores.items():
                        node_id = score_data['node_id']
                        miner_hotkey = score_data['miner_hotkey']
                        
                        logger.info(f"Processing response {response_id} for node {node_id}")
                        
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
                        
                        # Update external API for each response
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
                            started_at=(score_data['started_at']),
                            completed_at=(score_data['completed_at'])
                        )
                        
                        if update_success:
                            logger.info(f"Successfully updated API with scores for response {response_id}")
                        else:
                            logger.warning(f"Failed to update API with scores for response {response_id}")
                        
                        # Add a small delay between API calls to prevent rate limiting
                        await asyncio.sleep(0.5)
                    
                    logger.info(f"Completed processing all {len(scores)} responses for challenge {challenge['challenge_id']}")
            
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