#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from pathlib import Path
import time
from loguru import logger

# Add miner directory to Python path to match uvicorn's environment
miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

from utils.model_manager import ModelManager
from utils.video_downloader import download_video
from endpoints.soccer import process_soccer_video
from utils.device import get_optimal_device

# Test video URL
TEST_VIDEO_URL = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/test_001.mp4"

async def main():
    """Run a test video through the processing pipeline."""
    try:
        logger.info("Starting video processing test")
        start_time = time.time()
        
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
            
            # Save results
            output_dir = Path(__file__).parent.parent / "test_outputs"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"pipeline_test_results_{int(time.time())}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            # Print summary
            total_time = time.time() - start_time
            frames = len(result["frames"])
            fps = frames / result["processing_time"]
            
            logger.info("Processing completed successfully!")
            logger.info(f"Total frames processed: {frames}")
            logger.info(f"Processing time: {result['processing_time']:.2f} seconds")
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