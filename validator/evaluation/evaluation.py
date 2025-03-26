import os
import json
import base64
import httpx
import tempfile
from typing import Dict, List, Tuple, Optional
import random
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import asyncio
from openai import OpenAI
from fiber.logging_utils import get_logger
from validator.challenge.challenge_types import (
    GSRResponse,
    GSRChallenge,
    ValidationResult
)
from validator.config import FRAMES_TO_VALIDATE
from validator.evaluation.prompts import VALIDATION_PROMPT
from validator.utils.vlm_api import VLMProcessor
from validator.evaluation.bbox_clip import evaluate_frame

FRAME_TIMEOUT = 180.0  # seconds

logger = get_logger(__name__)

# Class IDs
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Colors
COLORS = {
    "player": (0, 255, 0),
    "goalkeeper": (0, 0, 255),
    "referee": (255, 0, 0),
    "ball": (0, 255, 255),
    "keypoint": (255, 0, 255)
}

def optimize_coordinates(coords: List[float]) -> List[float]:
    """Round coordinates to 2 decimals."""
    return [round(float(x), 2) for x in coords]

def filter_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """Remove zero-coord keypoints; round others to 2 decimals."""
    return [optimize_coordinates(kp) for kp in keypoints if not (kp[0] == 0 and kp[1] == 0)]

class GSRValidator:
    def __init__(self, openai_api_key: str, validator_hotkey: str):
        self.openai_api_key = openai_api_key
        self.validator_hotkey = validator_hotkey
        self.db_manager = None
        self._video_cache = {}
        self.vlm_processor = VLMProcessor(openai_api_key)

    def encode_image(self, image):
        """Base64-encode an image."""
        ok, buf = cv2.imencode('.jpg', image)
        return base64.b64encode(buf).decode('utf-8') if ok else ""

    async def download_video(self, video_url: str) -> Path:
        """
        Download video or return from cache if possible. Handles direct URLs or Google Drive.
        """
        if video_url in self._video_cache:
            cached_path = self._video_cache[video_url]
            if cached_path.exists():
                logger.info(f"Using cached video at: {cached_path}")
                return cached_path
            else:
                del self._video_cache[video_url]

        logger.info(f"Downloading video from: {video_url}")
        if 'drive.google.com' in video_url:
            file_id = None
            if 'id=' in video_url:
                file_id = video_url.split('id=')[1].split('&')[0]
            elif '/d/' in video_url:
                file_id = video_url.split('/d/')[1].split('/')[0]
            if not file_id:
                raise ValueError("Failed to extract Google Drive file ID from URL")
            video_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

        max_retries, retry_delay, timeout = 3, 5, 60.0
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.get(video_url)
                    if resp.status_code==404:
                        logger.error(f"❌ Video not found (404): {video_url}. Skipping challenge.")
                        return None
                    resp.raise_for_status()
                    temp_dir = Path(tempfile.gettempdir())
                    path = temp_dir / f"video_{datetime.now().timestamp()}.mp4"
                    path.write_bytes(resp.content)
                    if not path.exists() or path.stat().st_size == 0:
                        raise ValueError("Video is empty/missing")

                    cap = cv2.VideoCapture(str(path))
                    if not cap.isOpened():
                        cap.release()
                        path.unlink(missing_ok=True)
                        raise ValueError("Not a valid video")
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    logger.info(f"Video stats: {frame_count} frames, {fps} FPS")
                    self._video_cache[video_url] = path
                    return path
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All download attempts failed: {str(e)}")
                    if 'path' in locals() and path.exists():
                        path.unlink()
                    raise ValueError(f"Failed to download: {str(e)}")


    async def validate_keypoints(self, frame: np.ndarray, keypoints: list, frame_idx: int) -> float:
        """
        Validate keypoints. Uses batched VLM processor.
        Expects a numeric score from 0.0 to 1.0.
        """
        if not keypoints:
            logger.info(f"No keypoints to validate for frame {frame_idx}")
            return 0.0

        # Filter out zero coordinates and round to 2 decimals
        valid_keypoints = filter_keypoints(keypoints)
        if not valid_keypoints:
            logger.info(f"No valid keypoints after filtering for frame {frame_idx}")
            return 0.0

        #logger.info(f"Validating {len(valid_keypoints)} keypoints for frame {frame_idx}")

        kp_frame = frame.copy()
        for (x, y) in valid_keypoints:
                cv2.circle(kp_frame, (int(x), int(y)), 5, COLORS["keypoint"], -1)

        ref_path = Path(__file__).parent / "pitch-keypoints.jpg"
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            logger.error(f"Failed to load reference keypoint image from {ref_path}")
            return 0.0

        ref_encoded = self.encode_image(ref_img)
        kp_encoded = self.encode_image(kp_frame)
        
        if not (ref_encoded and kp_encoded):
            logger.error("Failed to encode reference or keypoint images")
            return 0.0

        frames_data = [{
            "reference_image": ref_encoded,
            "keypoint_image": kp_encoded,
            "frame_id": frame_idx
        }]

        try:
            logger.info(f"Sending keypoint validation request for frame {frame_idx}")
            results = await self.vlm_processor.validate_keypoints_batch(frames_data, VALIDATION_PROMPT)
            score = results[0] if results else 0.0
            logger.info(f"Keypoint validation score for frame {frame_idx}: {score}")
            return score
        except Exception as e:
            logger.error(f"Error validating keypoints for frame {frame_idx}: {str(e)}")
            return 0.0

    async def validate_bbox_clip(self, frame_idx: int, frame, detections: dict) -> float:
        try:
            objects = detections.get("objects", [])
            if not objects:
                return 0.0
            return evaluate_frame(frame_idx, frame.copy(), objects)
        except Exception as e:
            logger.error(f"[Frame {frame_idx}] BBox CLIP validation failed: {e}")
            return 0.0
        
    async def evaluate_response(
        self,
        response: GSRResponse,
        challenge: GSRChallenge,
        video_path: Path,
        frame_cache: Dict = None,
        frames_to_validate: List[int] = None
    ) -> ValidationResult:
        """
        Main entry to evaluate a GSR response.
        """
        if not getattr(response, 'response_id', None):
            raise ValueError("response_id is required")
        
        node_id = getattr(response, 'node_id', None)
        if node_id is None:
            raise ValueError("node_id is required")

        # Use provided frames or select new ones
        if frames_to_validate is None:
            frames_to_validate = self.select_random_frames(video_path)
        #logger.info(f"Evaluating frames: {frames_to_validate}")

        # Pre-fetch reference counts for all frames
        if frame_cache is None:
            frame_cache = {}
            
        frames_to_process = []
        for frame_idx in frames_to_validate:
            if frame_idx not in frame_cache:
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_cache[frame_idx] = {'frame': frame}
                    frames_to_process.append({
                        'encoded_image': self.encode_image(frame),
                        'frame_id': frame_idx
                    })


        tasks = []
        for frame_idx in frames_to_validate:
            try:
                frame = frame_cache[frame_idx]['frame']
                frame_data = response.frames.get(str(frame_idx), {})
                
                tasks.append((frame_idx, asyncio.create_task(
                    self.validate_frame_detections(
                        frame=frame,
                        detections=frame_data,
                        frame_idx=frame_idx,
                        challenge_id=challenge.challenge_id,
                        node_id=response.node_id,
                        response_id=response.response_id
                    )
                )))
            except Exception as e:
                logger.error(f"Error preparing frame {frame_idx}: {str(e)}")

        frame_evals = []
        for frame_idx, task in tasks:
            try:
                data = await task
                data["frame_number"] = frame_idx
                frame_evals.append(data)
                # Store in DB
                if self.db_manager:
                    scores = data["scores"]
                    feed = {
                        "scores": {
                            "keypoint_score": scores["keypoint_score"],
                            "bbox_score": scores["bbox_score"]
                        },
                        "scoring_details": data["scoring_details"]
                    }
                    self.db_manager.store_frame_evaluation(
                        response_id=response.response_id,
                        challenge_id=challenge.challenge_id,
                        miner_hotkey=response.miner_hotkey,
                        node_id=response.node_id,
                        frame_id=frame_idx,
                        frame_timestamp=frame_idx / 30.0,  # assume 30fps
                        frame_score=scores["final_score"],
                        raw_frame_path="",
                        annotated_frame_path=data["debug_frame_path"],
                        feedback=json.dumps(feed)
                    )
            except Exception as e:
                logger.error(f"Error evaluating frame {frame_idx}: {str(e)}")

        if not frame_evals:
            return ValidationResult(
                score=0.0,
                frame_scores={},
                feedback="No frames evaluated successfully."
            )

        # Gather results
        total_scores, frame_scores, details = [], {}, []
        for item in frame_evals:
            try:
                frm_num = item["frame_number"]
                final_score = item["scores"]["final_score"]
                total_scores.append(final_score)
                frame_scores[frm_num] = final_score
                
                # Extract scoring details safely
                frame_detail = {
                    "frame_number": frm_num,
                    "debug_frame_path": item.get("debug_frame_path", ""),
                    "scores": item.get("scores", {}),
                }
                
                # Only add scoring_details if it exists
                if "scoring_details" in item:
                    frame_detail["scoring_details"] = item["scoring_details"]
                    
                details.append(frame_detail)
            except Exception as e:
                logger.error(f"Error processing frame evaluation result for frame {frm_num}: {str(e)}")
                continue

        avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        summary = {
            "node_id": response.node_id,
            "challenge_id": challenge.challenge_id,
            "average_score": avg_score,
            "frame_count": len(frame_evals),
            "frame_details": details
        }
        #logger.info(f"Validation Results:\n{json.dumps(summary, indent=2)}")

        return ValidationResult(
            score=avg_score,
            frame_scores=frame_scores,
            feedback=details
        )

    def calculate_bbox_confidence_score(self, results: dict) -> float:
        """
        Weighted average of all object validation scores (0..1).
        Different classes get different weighting.
        """
        objs = results.get("objects", [])
        if not objs:
            return 0.0

        total, weight_sum = 0.0, 0.0
        weights = {
            "soccer ball": 0.7,
            "goalkeeper": 0.3,
            "referee": 0.2,
            "soccer player": 1.0
        }
        for o in objs:
            cls_name = o["class"]
            prob = o["probability"]
            w = weights.get(cls_name, 0.5)
            total += prob * w
            weight_sum += w
        return total / weight_sum if weight_sum else 0.0

    def calculate_final_score(self, keypoint_score: float, bbox_score: float) -> float:
        """
        Combine keypoints, bboxes, and object counts into final 0..1.
        """
        KEY_W, BOX_W = 0.5, 0.5
        return (
            (keypoint_score * KEY_W) +
            (bbox_score * BOX_W) 
        )
    

    def select_random_frames(self, video_path: Path, num_frames: int = None) -> List[int]:
        """
        Randomly pick frames from a video, skipping start/end buffer.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to select. If None, uses FRAMES_TO_VALIDATE from config
            
        Returns:
            List of selected frame numbers
        """
        num_frames = num_frames or FRAMES_TO_VALIDATE
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Calculate buffer size - either 30 frames or 10% of total, whichever is smaller
        buffer = min(5, total // 2)
        
        # Ensure we have enough frames to sample from
        if total <= (2 * buffer):
            logger.warning(f"Video too short ({total} frames) for buffer size {buffer}")
            buffer = total // 4  # Use 25% of total as buffer if video is very short
            
        # Calculate valid frame range
        start_frame = buffer
        end_frame = max(buffer, total - buffer)
        valid_range = range(start_frame, end_frame)
        
        if len(valid_range) < num_frames:
            logger.warning(f"Not enough frames ({len(valid_range)}) to select {num_frames} samples")
            num_frames = len(valid_range)
            
        frames = random.sample(valid_range, num_frames) if valid_range else []
        logger.info(f"Selected {len(frames)} frames from {total}")
        return sorted(frames)

    def draw_annotations(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Draw bounding boxes and keypoints onto the frame."""
        out = frame.copy()
        for obj in detections.get("objects", []):
            (x1, y1, x2, y2) = obj["bbox"]
            cid = obj["class_id"]
            if cid == BALL_CLASS_ID:
                color = COLORS["ball"]
            elif cid == GOALKEEPER_CLASS_ID:
                color = COLORS["goalkeeper"]
            elif cid == REFEREE_CLASS_ID:
                color = COLORS["referee"]
            else:
                color = COLORS["player"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        for (x, y) in detections.get("keypoints", []):
            if x != 0 and y != 0:
                cv2.circle(out, (int(x), int(y)), 5, COLORS["keypoint"], -1)
        return out

    def get_class_name(self, class_id: int) -> str:
        """Map class_id to string. (Legacy usage retained.)"""
        names = {0: "soccer ball", 1: "goalkeeper", 2: "player", 3: "referee"}
        return names.get(class_id, "unknown")

    def validate_bbox_coordinates(
        self,
        bbox: List[float],
        frame_shape: Tuple[int, int],
        class_id: int
    ) -> Optional[List[int]]:
        """
        Clamp bbox to frame bounds. Discard invalid or tiny ones.
        """
        try:
            h, w = frame_shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            if x2 <= x1 or y2 <= y1:
                return None
            x1, x2 = sorted([max(0, min(x1, w)), max(0, min(x2, w))])
            y1, y2 = sorted([max(0, min(y1, h)), max(0, min(y2, h))])
            if class_id == 0:  # ball can be small
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    return None
            else:
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    return None
            return [x1, y1, x2, y2]
        except Exception as e:
            logger.error(f"BBox validation error: {str(e)}")
            return None

    def resize_frame(self, frame: np.ndarray, target_width: int = 400) -> np.ndarray:
        """Keep aspect ratio on resize."""
        h, w = frame.shape[:2]
        aspect = w / h
        return cv2.resize(frame, (target_width, int(target_width / aspect)))

    def filter_detections(self, detections: Dict, shape: Tuple[int, int]) -> Dict:
        """Clamp bboxes, keep valid ones, preserve keypoints."""
        valid = {"objects": [], "keypoints": detections.get("keypoints", [])}
        for obj in detections.get("objects", []):
            bbox = self.validate_bbox_coordinates(obj["bbox"], shape, obj["class_id"])
            if bbox:
                valid["objects"].append({**obj, "bbox": bbox})
        return valid
