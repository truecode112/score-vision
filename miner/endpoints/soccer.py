import os
import json
import time
import threading
import tempfile
from typing import Optional, Dict, Any
import supervision as sv
from ultralytics import YOLO
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.dependencies import get_config
from sports.common.ball import BallTracker
from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration

logger = get_logger(__name__)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'miner/data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'miner/data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'miner/data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 20
CONFIG = SoccerPitchConfiguration()

# Keep only one lock for tracking miner availability
miner_lock = asyncio.Lock()

# Helper functions from main.py
def get_crops(frame: np.ndarray, detections: sv.Detections):
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def resolve_goalkeepers_team_id(players: sv.Detections, players_team_id: np.array, goalkeepers: sv.Detections) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

def save_tracking_data_to_json(output_path, data):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def download_video(url: str) -> Path:
    """Download video with retries and proper redirect handling."""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # First request to get the redirect
            response = await client.get(url)
            
            if "drive.google.com" in url:
                # For Google Drive, we need to handle the download URL specially
                if "drive.usercontent.google.com" in response.url.path:
                    download_url = str(response.url)
                else:
                    # If we got redirected to the Google Drive UI, construct the direct download URL
                    file_id = url.split("id=")[1].split("&")[0]
                    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
                
                # Make the actual download request
                response = await client.get(download_url)
            
            response.raise_for_status()
            
            # Create temp file with .mp4 extension
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_file.write(response.content)
            temp_file.close()
            
            logger.info(f"Video downloaded successfully to {temp_file.name}")
            return Path(temp_file.name)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading video: {str(e)}")
        logger.error(f"Response status code: {e.response.status_code}")
        logger.error(f"Response headers: {e.response.headers}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")

async def process_soccer_video(video_url: str, device: str = "cpu") -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    start_time = time.time()
    
    try:
        logger.info(f"Downloading video from {video_url}")
        video_path = await download_video(video_url)
        logger.info(f"Video downloaded to {video_path}")

        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Download models if they don't exist
        PLAYER_DETECTION_MODEL_PATH = data_dir / "football-player-detection.pt"
        BALL_DETECTION_MODEL_PATH = data_dir / "football-ball-detection.pt"
        KEYPOINT_DETECTION_MODEL_PATH = data_dir / "football-field-detection.pt"
        
        if not PLAYER_DETECTION_MODEL_PATH.exists():
            logger.info("Downloading player detection model...")
            # Add model download logic here
            raise HTTPException(status_code=500, detail="Player detection model not found. Please download the required models first.")
            
        # Load models
        logger.info(f"Loading models on device: {device}")
        player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
        pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
        ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

        # Process video frames
        video_info = sv.VideoInfo.from_video_path(video_path)
        frame_generator = sv.get_video_frames_generator(source_path=video_path)
        
        tracker = sv.ByteTrack()
        ball_tracker = BallTracker(buffer_size=20)
        
        tracking_data = {"frames": []}
        
        for frame_number, frame in enumerate(frame_generator):
            # Detect pitch
            pitch_result = pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # Detect players
            player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            # Detect ball
            ball_result = ball_detection_model(frame, imgsz=640, verbose=False)[0]
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
            
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")
    finally:
        # Cleanup temp file
        if 'video_path' in locals():
            try:
                os.unlink(video_path)
            except:
                pass

def factory_router() -> APIRouter:
    router = APIRouter()
    
    @router.post("/challenge")
    async def process_challenge(
        request: Request,
        config: Config = Depends(get_config)
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
                
                # Process the video
                try:
                    tracking_data = await process_soccer_video(video_url, device=config.device)
                except Exception as video_error:
                    logger.error(f"Error processing video: {str(video_error)}")
                    logger.exception("Full video processing error:")
                    raise HTTPException(status_code=500, detail=f"Video processing error: {str(video_error)}")
                
                # Prepare response
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
                
            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except Exception as e:
                logger.error(f"Error processing soccer challenge: {str(e)}")
                logger.exception("Full error traceback:")
                raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
            finally:
                logger.info("Releasing miner lock...")
    
    return router 