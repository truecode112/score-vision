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
from validator.evaluation.prompts import COUNT_PROMPT, VALIDATION_PROMPT
from validator.utils.vlm_api import VLMProcessor

# Add timeout constants
OPENAI_TIMEOUT = 30.0  # seconds
BATCH_TIMEOUT = 60.0   # seconds
FRAME_TIMEOUT = 180.0  # seconds
MAX_CONCURRENT_CALLS = 1
VLM_RATE_LIMIT = 1  # requests per second
VLM_BATCH_SIZE = 5   # number of images per batch

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
        self.frame_reference_counts = {}  # Cache for reference counts

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

    async def get_reference_counts(self, frame: np.ndarray) -> Dict:
        """
        Count key entities in the frame.
        Uses a system prompt, fallback handled by ask_vlm.
        """
        encoded = self.encode_image(frame)
        frames_data = [{
            "encoded_image": encoded,
            "frame_id": "single"
        }]
        
        results = await self.vlm_processor.get_reference_counts_batch(frames_data, COUNT_PROMPT)
        if not results or not results[0]:
            logger.warning("VLM returned empty content for reference counts")
            return {}
            
        try:
            cleaned_content = results[0]
            counts = json.loads(cleaned_content)
            if not isinstance(counts, dict):
                logger.warning(f"VLM response is not a dictionary: {counts}")
                return {}
                
            # Normalize the keys
            normalized = {}
            for key, value in counts.items():
                key = key.lower()
                if isinstance(value, (int, float)):
                    if "ball" in key or "soccer ball" in key:
                        normalized["soccer ball"] = int(value)
                    elif "goalkeeper" in key:
                        normalized["goalkeeper"] = int(value)
                    elif "referee" in key:
                        normalized["referee"] = int(value)
                    elif "player" in key:
                        normalized["player"] = int(value)
            
            logger.info(f"Normalized counts: {normalized}")
            return normalized
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse VLM response as JSON: {cleaned_content}")
            return {}
            
        except Exception as e:
            logger.error(f"Reference count error: {str(e)}")
            return {}

    async def validate_bbox_content_batch(self, images: List[np.ndarray], expected_class: str) -> List[float]:
        """
        Validate multiple bounding boxes in one call with rate limiting.
        """
        if not images:
            return []

        # Prepare batch data
        batch_data = []
        for i, img in enumerate(images):
            encoded = self.encode_image(img)
            if encoded:
                batch_data.append({
                    "encoded_image": encoded,
                    "bbox_id": i
                })

        return await self.vlm_processor.validate_bbox_content_batch(batch_data, expected_class)

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

    async def validate_frame_detections(
        self,
        frame: np.ndarray,
        detections: dict,
        frame_idx: int,
        challenge_id: str,
        node_id: int,
        reference_counts: Dict = None,
        response_id: str = None
    ) -> dict:
        """
        Validate detections in a single frame with timeout protection.
        """
        # Initialize results with default values
        results = {
            "objects": [],
            "keypoints": {"score": 0.0, "points": [], "visualization_path": ""},
            "scores": {
                "keypoint_score": 0.0,
                "bbox_score": 0.0,
                "count_match_score": 0.0,
                "final_score": 0.0
            },
            "debug_frame_path": "",
            "timing": {}
        }

        try:
            async def _process_frame():
                start = datetime.now()
                logger.info(f"Validating frame {frame_idx} (challenge {challenge_id}, node {node_id})")

                # Run validations concurrently
                filtered = self.filter_detections(detections, frame.shape)
                kpts = filtered.get("keypoints", [])
                #logger.info(f"Processing {len(kpts)} keypoints for frame {frame_idx}")
                kpt_score_task = self.validate_keypoints(frame, kpts, frame_idx)

                # Process objects first
                class_map = {}
                for obj in filtered.get("objects", []):
                    cls = self.get_class_name(obj["class_id"])
                    x1, y1, x2, y2 = obj["bbox"]
                    crop = frame[y1:y2, x1:x2]
                    if cls not in class_map:
                        class_map[cls] = {"images": [], "objs": []}
                    class_map[cls]["images"].append(crop)
                    class_map[cls]["objs"].append(obj)

                # Process each class with rate limiting
                obj_scores = []
                for cls, group in class_map.items():
                    confs = await self.validate_bbox_content_batch(group["images"], cls)
                    for ob, c in zip(group["objs"], confs):
                        obj_scores.append({"class": cls, "score": c})
                        results["objects"].append({
                            "bbox_idx": ob["id"],
                            "class": cls,
                            "class_id": ob["class_id"],
                            "probability": c
                        })

                # Calculate scores
                box_score = self.calculate_bbox_confidence_score(results)
                
                # Get reference counts - use cached version if available
                ref_counts = None
                if frame_idx in self.frame_reference_counts:
                    ref_counts = self.frame_reference_counts[frame_idx]
                elif reference_counts:
                    ref_counts = reference_counts
                else:
                    # Only fetch if absolutely necessary
                    logger.warning(f"No cached reference counts for frame {frame_idx}, fetching new...")
                    ref_counts = await self.get_reference_counts(frame)
                    self.frame_reference_counts[frame_idx] = ref_counts
                    
                kpt_score = await kpt_score_task
                results["keypoints"] = {
                    "score": kpt_score,
                    "points": kpts,
                    "visualization_path": ""
                }

                count_val = self.compare_with_reference_counts(ref_counts, results)
                final_score = self.calculate_final_score(kpt_score, box_score, count_val["match_score"])

                # Save debug visualization
                ann = self.draw_annotations(frame, filtered)
                ann_resized = self.resize_frame(ann, target_width=400)

                current_date = datetime.now().strftime("%Y%m%d")
                dbg_dir = Path("debug_frames") / current_date
                dbg_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Include response_id in filename if available
                filename_parts = [
                    f"challenge_{challenge_id}",
                    f"node_{node_id}",
                    f"frame_{frame_idx}"
                ]
                if response_id:
                    filename_parts.insert(1, f"resp_{response_id}")  # Insert after challenge_id
                    
                dbg_path = dbg_dir / f"{('_'.join(filename_parts))}_{ts}.jpg"
                cv2.imwrite(str(dbg_path), ann_resized)

                # Update results
                results["scores"].update({
                    "keypoint_score": kpt_score,
                    "bbox_score": box_score,
                    "count_match_score": count_val["match_score"],
                    "final_score": final_score
                })
                results["debug_frame_path"] = str(dbg_path)
                results["count_validation"] = count_val
                results["scoring_details"] = {
                    "keypoint_score": {"score": kpt_score, "weight": 0.4},
                    "bbox_score": {"score": box_score, "weight": 0.4, "object_scores": obj_scores},
                    "count_match_score": {
                        "score": count_val["match_score"],
                        "weight": 0.2,
                        "reference_counts": ref_counts,
                        "detected_counts": count_val["high_confidence_counts"]
                    }
                }
                results["timing"] = {
                    "total_time": (datetime.now() - start).total_seconds()
                }

            await asyncio.wait_for(_process_frame(), timeout=FRAME_TIMEOUT)

        except asyncio.TimeoutError:
            logger.error(f"Frame {frame_idx} validation timed out")
        except Exception as e:
            logger.error(f"Frame {frame_idx} validation error: {str(e)}")

        return results

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

        if frames_to_process:
            reference_results = await self.vlm_processor.get_reference_counts_batch(frames_to_process, COUNT_PROMPT)
            for frame_data, result in zip(frames_to_process, reference_results):
                frame_idx = frame_data['frame_id']
                try:
                    if result:
                        counts = json.loads(result)
                        if isinstance(counts, dict):
                            normalized = {}
                            for key, value in counts.items():
                                key = key.lower()
                                if isinstance(value, (int, float)):
                                    if "ball" in key:
                                        normalized["ball"] = int(value)
                                    elif "goalkeeper" in key:
                                        normalized["goalkeeper"] = int(value)
                                    elif "referee" in key:
                                        normalized["referee"] = int(value)
                                    elif "player" in key:
                                        normalized["player"] = int(value)
                            self.frame_reference_counts[frame_idx] = normalized
                except Exception as e:
                    logger.error(f"Error processing reference counts for frame {frame_idx}: {str(e)}")

        tasks = []
        for frame_idx in frames_to_validate:
            try:
                frame = frame_cache[frame_idx]['frame']
                frame_data = response.frames.get(str(frame_idx), {})
                ref_counts = self.frame_reference_counts.get(frame_idx)
                
                tasks.append((frame_idx, asyncio.create_task(
                    self.validate_frame_detections(
                        frame=frame,
                        detections=frame_data,
                        frame_idx=frame_idx,
                        challenge_id=challenge.challenge_id,
                        node_id=response.node_id,
                        reference_counts=ref_counts,
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
                            "bbox_score": scores["bbox_score"],
                            "count_match_score": scores["count_match_score"]
                        },
                        "reference_counts": data["count_validation"]["reference_counts"],
                        "detected_counts": data["count_validation"]["high_confidence_counts"],
                        "count_matches": data["count_validation"]["count_matches"],
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
                        vlm_response=data["count_validation"]["reference_counts"],
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

    def calculate_final_score(self, keypoint_score: float, bbox_score: float, count_match: float) -> float:
        """
        Combine keypoints, bboxes, and object counts into final 0..1.
        """
        KEY_W, BOX_W, CNT_W = 0.4, 0.4, 0.2
        return (
            (keypoint_score * KEY_W) +
            (bbox_score * BOX_W) +
            (count_match * CNT_W)
        )

    def compare_with_reference_counts(self, ref_counts: Dict[str, int], results: Dict) -> Dict:
        """
        Compare the number of high-confidence detections vs. reference counts.
        Returns normalized match scores per class and overall match score.
        """
        # Normalize reference counts first
        normalized_ref = {
            "player": 0,
            "goalkeeper": 0,
            "referee": 0,
            "ball": 0
        }
        
        if ref_counts:
            for key, value in ref_counts.items():
                key = key.lower()
                if isinstance(value, (int, float)):
                    if "ball" in key or "soccer ball" in key:
                        normalized_ref["ball"] = int(value)
                    elif "goalkeeper" in key:
                        normalized_ref["goalkeeper"] = int(value)
                    elif "referee" in key:
                        normalized_ref["referee"] = int(value)
                    elif "player" in key:
                        normalized_ref["player"] = int(value)

        # Count high confidence detections (prob >= 0.7)
        high_conf = {
            "player": 0,
            "goalkeeper": 0,
            "referee": 0,
            "ball": 0
        }

        for obj in results.get("objects", []):
            if obj.get("probability", 0) >= 0.7:
                cls = obj["class"].lower()
                if "ball" in cls:
                    high_conf["ball"] += 1
                elif "goalkeeper" in cls:
                    high_conf["goalkeeper"] += 1
                elif "referee" in cls:
                    high_conf["referee"] += 1
                elif "player" in cls:
                    high_conf["player"] += 1

        # Calculate match scores with importance weights
        weights = {
            "player": 0.4,    # Players are most important
            "goalkeeper": 0.3, # Goalkeeper is important for team composition
            "referee": 0.2,   # Referee is less critical but should be counted
            "ball": 0.1      # Ball count is least important (usually just 1)
        }

        matches = {}
        weighted_sum = 0
        total_weight = 0

        for cls in ["player", "goalkeeper", "referee", "ball"]:
            ref = normalized_ref[cls]
            detected = high_conf[cls]
            
            # Calculate match score for this class
            if ref == 0 and detected == 0:
                matches[cls] = 1.0  # Perfect match when both are 0
            elif ref == 0:
                matches[cls] = 0.0  # Penalize false positives
            else:
                # Calculate ratio and apply penalties
                ratio = min(detected, ref) / ref
                # Penalize overcounting more than undercounting
                if detected > ref:
                    excess = (detected - ref) / ref
                    penalty = min(1.0, excess * 0.5)  # 50% penalty per excess count
                    ratio = max(0.0, ratio - penalty)
                matches[cls] = max(0.0, min(1.0, ratio))

            weight = weights[cls]
            weighted_sum += matches[cls] * weight
            total_weight += weight

        # Calculate final match score
        match_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Log detailed information for debugging
        logger.info("Count matching details:")
        logger.info(f"Reference counts: {normalized_ref}")
        logger.info(f"Detected counts: {high_conf}")
        logger.info(f"Individual matches: {matches}")
        logger.info(f"Final match score: {match_score}")

        return {
            "reference_counts": normalized_ref,
            "high_confidence_counts": high_conf,
            "count_matches": matches,
            "match_score": match_score
        }

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