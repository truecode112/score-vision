import os
import json
import time
from typing import Optional, Dict, Any
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.dependencies import get_config
from sports.common.ball import BallTracker
from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global model manager instance
model_manager = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager

async def process_soccer_video(
    video_path: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    start_time = time.time()
    
    try:
        # Initialize video processor with the same device as model manager
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,  # 3 hours max for any device
            mps_timeout=10800.0,   # We'll use the same timeout for all devices
            cpu_timeout=10800.0    # to ensure complete processing
        )
        
        # Verify video is readable
        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )
        
        # Get models from manager
        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        ball_model = model_manager.get_model("ball")
        
        # Initialize trackers
        tracker = sv.ByteTrack()
        ball_tracker = BallTracker(buffer_size=20)
        
        tracking_data = {"frames": []}
        
        # Process all frames
        async for frame_number, frame in video_processor.stream_frames(video_path):
            # Process frame with models - hardware acceleration will be used if available
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            ball_result = ball_model(frame, imgsz=640, verbose=False)[0]
            ball_detections = sv.Detections.from_ultralytics(ball_result)
            ball_detections = ball_tracker.update(ball_detections)
            
            # Process frame data
            frame_data = {
                "frame_number": frame_number,
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                "players": [
                    {
                        "id": int(tracker_id),
                        "bbox": bbox.tolist(),
                        "class_id": int(class_id)
                    }
                    for tracker_id, bbox, class_id in zip(
                        detections.tracker_id,
                        detections.xyxy,
                        detections.class_id
                    )
                ] if detections and detections.tracker_id is not None else [],
                "ball": [
                    {
                        "id": int(tracker_id),
                        "bbox": bbox.tolist()
                    }
                    for tracker_id, bbox in zip(
                        ball_detections.tracker_id,
                        ball_detections.xyxy
                    )
                ] if ball_detections and ball_detections.tracker_id is not None else []
            }
            tracking_data["frames"].append(frame_data)
            
            # Log progress every 100 frames
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")
        
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        # Log final statistics
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps:.2f} fps) on {model_manager.device} device"
        )
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

# Create router instance
router = APIRouter()

@router.post("/challenge")
async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Process a soccer challenge."""
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            # Get challenge data from request
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            
            # Download video
            video_path = await download_video(video_url)
            
            try:
                # Process the video
                tracking_data = await process_soccer_video(
                    video_path,
                    model_manager
                )
                
                # Prepare response
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
                
            finally:
                # Cleanup temp file
                try:
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing miner lock...")