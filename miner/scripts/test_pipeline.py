#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from pathlib import Path
import time
from loguru import logger
from typing import List, Dict, Union

# Add miner directory to Python path to match uvicorn's environment
miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager
from utils.video_downloader import download_video
from endpoints.soccer import process_soccer_video
from utils.device import get_optimal_device
from scripts.download_models import download_models

# Test video URL
TEST_VIDEO_URL = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/test_10secs.mov"

def optimize_coordinates(coords: List[float]) -> List[float]:
    """Round coordinates to 2 decimal places to reduce data size."""
    return [round(float(x), 2) for x in coords]

def filter_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """Filter out keypoints with zero coordinates and round remaining to 2 decimal places."""
    return [optimize_coordinates(kp) for kp in keypoints if not (kp[0] == 0 and kp[1] == 0)]

def optimize_frame_data(frame_data: Dict) -> Dict:
    """Optimize frame data by rounding coordinates and filtering zero keypoints."""
    optimized_data = {}
    
    # Optimize players data
    if "players" in frame_data:
        optimized_data["players"] = []
        for player in frame_data["players"]:
            optimized_player = player.copy()
            if "bbox" in player:
                optimized_player["bbox"] = optimize_coordinates(player["bbox"])
            optimized_data["players"].append(optimized_player)
    
    # Optimize ball data
    if "ball" in frame_data:
        optimized_data["ball"] = []
        for ball in frame_data["ball"]:
            optimized_ball = ball.copy()
            if "bbox" in ball:
                optimized_ball["bbox"] = optimize_coordinates(ball["bbox"])
            optimized_data["ball"].append(optimized_ball)
    
    # Optimize keypoints
    if "keypoints" in frame_data:
        optimized_data["keypoints"] = filter_keypoints(frame_data["keypoints"])
    
    return optimized_data

def optimize_result_data(result: Dict[str, Union[Dict, List, float, str]]) -> Dict[str, Union[Dict, List, float, str]]:
    """Optimize the entire result data, handling both list and dictionary frame formats."""
    optimized_result = result.copy()
    
    # Handle frames data
    if "frames" in result:
        frames = result["frames"]
        
        # If frames is a list, convert to dictionary with frame indices as keys
        if isinstance(frames, list):
            optimized_frames = {}
            for i, frame_data in enumerate(frames):
                if frame_data:  # Only include non-empty frames
                    optimized_frames[str(i)] = optimize_frame_data(frame_data)
        # If frames is already a dictionary
        elif isinstance(frames, dict):
            optimized_frames = {}
            for frame_num, frame_data in frames.items():
                if frame_data:  # Only include non-empty frames
                    optimized_frames[str(frame_num)] = optimize_frame_data(frame_data)
        else:
            logger.warning(f"Unexpected frames data type: {type(frames)}")
            optimized_frames = frames
            
        optimized_result["frames"] = optimized_frames
    
    # Round processing time if present
    if "processing_time" in result:
        optimized_result["processing_time"] = round(float(result["processing_time"]), 2)
    
    return optimized_result

async def main():
    """Run a test video through the processing pipeline."""
    try:
        logger.info("Starting video processing test")
        start_time = time.time()
        
        # Ensure models are downloaded
        logger.info("Checking for required models...")
        download_models()
        
        # Download test video
        logger.info(f"Downloading test video from {TEST_VIDEO_URL}")
        video_path = await download_video(TEST_VIDEO_URL)
        logger.info(f"Video downloaded to {video_path}")
        
        try:
            # Initialize model manager with auto-detected device
            device = get_optimal_device()
            logger.info(f"Using device: {device}")
            
            model_manager = ModelManager(device=device)
            
            # Load all models
            logger.info("Loading models...")
            model_manager.load_all_models()
            logger.info("Models loaded successfully")
            
            # Process the video
            logger.info("Starting video processing...")
            result = await process_soccer_video(
                video_path=str(video_path),
                model_manager=model_manager
            )
            
            # Optimize frame data
            logger.info("Optimizing frame data...")
            optimized_result = optimize_result_data(result)
            
            # Save results
            output_dir = Path(__file__).parent.parent / "test_outputs"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"pipeline_test_results_{int(time.time())}.json"
            
            # Log data size before saving
            result_json = json.dumps(optimized_result)
            data_size = len(result_json) / 1024  # Size in KB
            logger.info(f"Result data size: {data_size:.2f} KB")
            
            with open(output_file, "w") as f:
                f.write(result_json)
            
            # Print summary
            total_time = time.time() - start_time
            frames = len(optimized_result["frames"])
            fps = frames / optimized_result["processing_time"]
            
            logger.info("Processing completed successfully!")
            logger.info(f"Total frames processed: {frames}")
            logger.info(f"Processing time: {optimized_result['processing_time']:.2f} seconds")
            logger.info(f"Average FPS: {fps:.2f}")
            logger.info(f"Total time (including download): {total_time:.2f} seconds")
            logger.info(f"Results saved to: {output_file}")
            
        finally:
            # Clear model cache
            model_manager.clear_cache()
            
    finally:
        # Cleanup downloaded video
        try:
            video_path.unlink()
            logger.info("Cleaned up temporary video file")
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(main()) 