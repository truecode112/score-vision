import os
import json
import base64
import httpx
import tempfile
from typing import Dict, List, Tuple
import random
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from fiber.logging_utils import get_logger
from validator.challenge.challenge_types import (
    GSRResponse,
    GSRChallenge,
    ValidationResult
)
from validator.config import FRAMES_TO_VALIDATE
from validator.evaluation.prompts import COUNT_PROMPT, VALIDATION_PROMPT

logger = get_logger(__name__)

class GSRValidator:
    def __init__(self, openai_api_key: str, validator_hotkey: str = None):
        """Initialize validator with OpenAI API key for frame validation"""
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for frame validation")
        self.client = OpenAI(api_key=openai_api_key)
        self.validator_hotkey = validator_hotkey
        self.labels = ["player", "goalkeeper", "referee", "ball"]
        self._video_cache = {}  # Cache of downloaded videos
        logger.info("GSRValidator initialized - will perform video frame validation")

    def select_random_frames(self, video_path: Path, num_frames: int = None) -> List[int]:
        """Select random frames from video"""
        if num_frames is None:
            num_frames = FRAMES_TO_VALIDATE
            
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Select random frames, ensuring they're not too close to start/end
        buffer = min(30, total_frames // 10)  # 10% or 30 frames buffer
        frame_range = range(buffer, total_frames - buffer)
        frames = random.sample(frame_range, min(num_frames, len(frame_range)))
        frames.sort()  # Keep frames in order
        return frames
    
    def draw_bounding_boxes(self, frame: np.ndarray, frame_data: Dict) -> np.ndarray:
        """Draw bounding boxes and keypoints on frame based on miner's response"""
        try:
            annotated_frame = frame.copy()

            # Color mapping for different object types (BGR format)
            colors = {
                "player": (0, 255, 0),      # Green
                "goalkeeper": (0, 0, 255),   # Red
                "referee": (255, 0, 0),      # Blue
                "ball": (0, 255, 255),       # Yellow
                "keypoint": (255, 0, 255)    # Bright pink
            }

            # Handle different frame data formats
            if isinstance(frame_data, dict):
                # Draw pitch keypoints
                if "keypoints" in frame_data and frame_data["keypoints"]:
                    try:
                        keypoints = frame_data["keypoints"]
                        # Draw each keypoint as a circle with bright pink color
                        for kp in keypoints:
                            x, y = int(kp[0]), int(kp[1])
                            # Skip drawing if keypoint is 0,0
                            if x == 0 and y == 0:
                                continue
                            # Draw a larger filled circle
                            cv2.circle(annotated_frame, (x, y), 6, colors["keypoint"], -1)
                            # Draw a white border for better visibility
                            cv2.circle(annotated_frame, (x, y), 6, (255, 255, 255), 1)
                    except Exception as e:
                        logger.error(f"Error drawing keypoints: {str(e)}")

                # Draw players with different colors based on class
                if "players" in frame_data:
                    for player in frame_data["players"]:
                        try:
                            if "bbox" in player and "class_id" in player:
                                x1, y1, x2, y2 = player["bbox"]
                                class_id = player["class_id"]
                                # Map class_id to type
                                if class_id == 1:  # GOALKEEPER_CLASS_ID
                                    color = colors["goalkeeper"]
                                elif class_id == 3:  # REFEREE_CLASS_ID
                                    color = colors["referee"]
                                else:  # PLAYER_CLASS_ID
                                    color = colors["player"]
                                
                                cv2.rectangle(annotated_frame, 
                                            (int(x1), int(y1)), 
                                            (int(x2), int(y2)), 
                                            color, 2)
                        except Exception as e:
                            logger.error(f"Error drawing player bbox: {str(e)}")
                            continue

                # Draw ball
                if "ball" in frame_data:
                    for ball in frame_data["ball"]:
                        try:
                            if "bbox" in ball:
                                x1, y1, x2, y2 = ball["bbox"]
                                cv2.rectangle(annotated_frame, 
                                            (int(x1), int(y1)), 
                                            (int(x2), int(y2)), 
                                            colors["ball"], 2)
                        except Exception as e:
                            logger.error(f"Error drawing ball bbox: {str(e)}")
                            continue

            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error in draw_bounding_boxes: {str(e)}")
            # Return original frame if we can't draw boxes
            return frame.copy()

    def get_reference_counts(self, frame: np.ndarray) -> Dict:
        """Get reference counts of objects in frame using VLM"""
        # Encode frame for API
        encoded = self.encode_image(frame)
        
        # Create message array
        messages = [
            {"role": "system", "content": "You are an expert at counting objects in soccer match frames."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": COUNT_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}}
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                temperature=0.2
            )
            
            # Get the response content and clean it up
            content = response.choices[0].message.content.strip()
            
            # Remove any markdown code block indicators if present
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            try:
                result = json.loads(content)
                logger.info(f"Reference counts: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                logger.error(f"JSON parse error: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting reference counts: {str(e)}")
            return None

    def evaluate_frame(self, frame: np.ndarray, frame_data: Dict, reference_counts: Dict) -> Dict:
        """Evaluate a single frame's annotations"""
        # Draw bounding boxes on frame
        annotated_frame = self.draw_bounding_boxes(frame, frame_data)
        encoded = self.encode_image(annotated_frame)
        
        # Format validation prompt with reference counts
        formatted_validation_prompt = VALIDATION_PROMPT.format(
            reference_counts.get('player', 0),
            reference_counts.get('goalkeeper', 0),
            reference_counts.get('referee', 0),
            reference_counts.get('soccer ball', 0)
        )
        
        messages = [
            {"role": "system", "content": "You are an expert at evaluating bounding box annotations in soccer match frames."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_validation_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}}
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.2
            )
            
            # Get the response content and clean it up
            content = response.choices[0].message.content.strip()
            
            # Remove any markdown code block indicators if present
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            try:
                result = json.loads(content)
                logger.info(f"Frame evaluation result: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                logger.error(f"JSON parse error: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error evaluating frame: {str(e)}")
            return None

    async def evaluate_frames(self, video_path: Path, chosen_frames: List[int], frame_data: Dict) -> List[Dict]:
        """Evaluate all chosen frames from video in batch"""
        evaluations = []
        
        cap = cv2.VideoCapture(str(video_path))
        for frame_num in chosen_frames:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Failed to read frame {frame_num}, skipping")
                    continue
                
                # First pass: Get reference counts from raw frame
                reference_counts = self.get_reference_counts(frame)
                if reference_counts is None:
                    logger.error(f"Failed to get reference counts for frame {frame_num}, skipping")
                    continue
                
                # Second pass: Evaluate annotations using reference counts
                evaluation = self.evaluate_frame(frame, frame_data.get(str(frame_num), {}), reference_counts)
                if evaluation is None:
                    logger.error(f"Failed to evaluate frame {frame_num}, skipping")
                    continue
                
                # Add frame number and reference counts to result
                evaluation['frame_number'] = frame_num
                evaluation['reference_counts'] = reference_counts
                evaluations.append(evaluation)
                
                logger.info(f"Completed evaluation for frame {frame_num}")
            except Exception as e:
                logger.error(f"Error processing frame {frame_num}, skipping: {str(e)}")
                continue
            
        cap.release()
        
        if not evaluations:
            logger.warning("No frames were successfully evaluated")
            return []
            
        return evaluations

    async def evaluate_response(
        self, 
        response: GSRResponse, 
        challenge: GSRChallenge,
        video_path: Path
    ) -> ValidationResult:
        """Evaluate a GSR response"""
        try:
            if not hasattr(response, 'response_id') or response.response_id is None:
                logger.error("Response object missing response_id")
                raise ValueError("Response object must have response_id set")
                
            # Select random frames to evaluate
            chosen_frames = self.select_random_frames(video_path)
            logger.info(f"Selected {len(chosen_frames)} frames for validation: {chosen_frames}")
            
            # Create debug frames directory
            debug_dir = Path("debug_frames")
            debug_dir.mkdir(exist_ok=True)
            
            # Pick one random frame to save as representative sample
            sample_frame = random.choice(chosen_frames)
            logger.info(f"Selected frame {sample_frame} as representative sample")
            
            # Evaluate all frames in batch
            frame_evaluations = await self.evaluate_frames(video_path, chosen_frames, response.frames)
            
            if not frame_evaluations:
                logger.error("No frames were successfully evaluated")
                return ValidationResult(
                    score=0.0,
                    frame_scores={},
                    feedback="Failed to evaluate any frames successfully"
                )
            
            logger.info(f"Successfully evaluated {len(frame_evaluations)} frames")
            
            # Process evaluations
            scores = []
            feedbacks = []
            frame_scores = {}
            
            for eval_data in frame_evaluations:
                frame_num = eval_data["frame_number"]
                score = eval_data["accuracy_score"]
                feedback = "\n".join(eval_data["discrepancies"])
                reference_counts = eval_data["reference_counts"]
                annotation_counts = eval_data["annotation_counts"]
                
                # Save debug frames only for the sample frame
                if frame_num == sample_frame:
                    # Save annotated frame with challenge and miner info in filename
                    sample_path = debug_dir / f"challenge_{challenge.challenge_id}_frame_{frame_num}_miner_{response.miner_hotkey}.jpg"
                    annotated_frame = self.draw_bounding_boxes(
                        self.extract_frame(video_path, frame_num), 
                        response.frames.get(str(frame_num), {})
                    )
                    cv2.imwrite(str(sample_path), annotated_frame)
                    logger.info(f"Saved sample annotated frame to {sample_path}")
                
                # Store frame evaluation in database
                if hasattr(self, 'db_manager'):
                    self.db_manager.store_frame_evaluation(
                        response_id=response.response_id,
                        challenge_id=challenge.challenge_id,
                        miner_hotkey=response.miner_hotkey,
                        node_id=response.node_id,
                        frame_id=frame_num,
                        frame_timestamp=frame_num / 30.0,  # Assuming 30fps
                        frame_score=score,
                        raw_frame_path=str(sample_path) if frame_num == sample_frame else None,
                        annotated_frame_path=str(sample_path) if frame_num == sample_frame else None,
                        vlm_response=reference_counts,
                        feedback=feedback
                    )
                    logger.info(f"Stored evaluation for frame {frame_num} in database")
                
                scores.append(score)
                feedbacks.append(f"Frame {frame_num}: {feedback}")
                frame_scores[frame_num] = score
            
            # Calculate final score and combine feedback
            avg_score = sum(scores) / len(scores) if scores else 0.0
            combined_feedback = "\n".join(feedbacks)
            
            logger.info(f"Validation complete. Average score: {avg_score:.3f}")
            return ValidationResult(
                score=avg_score,
                frame_scores=frame_scores,
                feedback=combined_feedback
            )
            
        except Exception as e:
            error_msg = f"Error evaluating response {response.challenge_id}: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                score=0.0,
                frame_scores={},
                feedback=error_msg
            )

    async def download_video(self, video_url: str) -> Path:
        """Download video to temporary file, handling redirects and Google Drive URLs"""
        # Check cache first
        if video_url in self._video_cache:
            cached_path = self._video_cache[video_url]
            if cached_path.exists():
                logger.info(f"Using cached video from: {cached_path}")
                return cached_path
            else:
                # Remove from cache if file no longer exists
                del self._video_cache[video_url]

        logger.info(f"Downloading video from: {video_url}")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Handle Google Drive URLs
            if 'drive.google.com' in video_url:
                logger.debug("Detected Google Drive URL, extracting file ID...")
                # Extract file ID
                file_id = None
                if 'id=' in video_url:
                    file_id = video_url.split('id=')[1].split('&')[0]
                elif '/d/' in video_url:
                    file_id = video_url.split('/d/')[1].split('/')[0]
                
                if not file_id:
                    raise ValueError("Could not extract Google Drive file ID from URL")
                
                # Use the direct download URL
                video_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
                logger.debug(f"Using direct download URL: {video_url}")

            try:
                logger.debug("Starting video download...")
                response = await client.get(video_url)
                response.raise_for_status()

                # Save to temporary file
                temp_dir = Path(tempfile.gettempdir())
                video_path = temp_dir / f"video_{datetime.now().timestamp()}.mp4"
                video_path.write_bytes(response.content)
                logger.info(f"Video downloaded to: {video_path}")
                
                # Verify the downloaded file
                if not video_path.exists() or video_path.stat().st_size == 0:
                    raise ValueError("Downloaded video file is empty or does not exist")
                
                # Try opening with OpenCV to verify it's a valid video
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    cap.release()
                    video_path.unlink(missing_ok=True)
                    raise ValueError("Downloaded file is not a valid video")
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                logger.info(f"Valid video file: {frame_count} frames, {fps} FPS")
                
                # Cache the video path
                self._video_cache[video_url] = video_path
                
                return video_path
                
            except httpx.HTTPError as e:
                raise ValueError(f"Failed to download video: {str(e)}")
            except Exception as e:
                raise ValueError(f"Error downloading video: {str(e)}")

    def extract_frame(self, video_path: Path, frame_num: int) -> np.ndarray:
        """Extract frame from video at given frame number"""
        logger.debug(f"Extracting frame {frame_num} from {video_path}")
        cap = cv2.VideoCapture(str(video_path))

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 0-based index
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not extract frame {frame_num}")
        logger.debug(f"Successfully extracted frame {frame_num}")
        return frame

    def encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string"""
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode('utf-8')

    async def validate_response(
        self, 
        response: GSRResponse, 
        challenge: GSRChallenge,
        video_path: Path
    ) -> ValidationResult:
        """Validate GSR response against challenge"""
        logger.info(f"Starting validation of response for challenge {challenge.challenge_id}")
        
        try:
            # Create debug frames directory
            debug_dir = Path("debug_frames")
            debug_dir.mkdir(exist_ok=True)
            
            # Get frames to validate
            chosen_frames = self.select_random_frames(video_path)
            logger.info(f"Selected {len(chosen_frames)} frames for validation: {chosen_frames}")
            
            # Evaluate all frames in batch
            frame_evaluations = await self.evaluate_frames(video_path, chosen_frames, response.frames)
            
            if not frame_evaluations:
                logger.error("No frames were successfully evaluated")
                return ValidationResult(
                    score=0.0,
                    frame_scores={},
                    feedback="Failed to evaluate any frames successfully"
                )
            
            logger.info(f"Successfully evaluated {len(frame_evaluations)} frames")
            
            # Process evaluations
            scores = []
            feedbacks = []
            frame_scores = {}
            
            for eval_data in frame_evaluations:
                frame_num = eval_data["frame_number"]
                score = eval_data["accuracy_score"]
                feedback = "\n".join(eval_data["discrepancies"])
                reference_counts = eval_data["reference_counts"]
                annotation_counts = eval_data["annotation_counts"]
                
                # Save debug frames
                raw_path = debug_dir / f"frame_{frame_num}_raw.jpg"
                annotated_path = debug_dir / f"frame_{frame_num}_annotated.jpg"
                cv2.imwrite(str(raw_path), self.extract_frame(video_path, frame_num))
                cv2.imwrite(str(annotated_path), self.draw_bounding_boxes(self.extract_frame(video_path, frame_num), response.frames.get(str(frame_num), {})))
                logger.info(f"Saved debug frames for frame {frame_num}")
                
                # Store frame evaluation in database
                if hasattr(self, 'db_manager'):
                    self.db_manager.store_frame_evaluation(
                        response_id=response.response_id,  # Changed from challenge_id to response_id
                        challenge_id=challenge.challenge_id,
                        miner_hotkey=response.miner_hotkey,
                        node_id=response.node_id,
                        frame_id=frame_num,
                        frame_timestamp=frame_num / 30.0,  # Assuming 30fps
                        frame_score=score,
                        raw_frame_path=str(raw_path),
                        annotated_frame_path=str(annotated_path),
                        vlm_response=reference_counts,
                        feedback=feedback
                    )
                    logger.info(f"Stored evaluation for frame {frame_num} in database")
                
                scores.append(score)
                feedbacks.append(f"Frame {frame_num}: {feedback}")
                frame_scores[frame_num] = score
            
            # Calculate final score and combine feedback
            avg_score = sum(scores) / len(scores) if scores else 0.0
            combined_feedback = "\n".join(feedbacks)
            
            logger.info(f"Validation complete. Average score: {avg_score:.3f}")
            return ValidationResult(
                score=avg_score,
                frame_scores=frame_scores,
                feedback=combined_feedback
            )
            
        except Exception as e:
            error_msg = f"Error during validation: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                score=0.0,
                frame_scores={},
                feedback=error_msg
            )