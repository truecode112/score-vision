from enum import Enum, auto
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import asyncio

class ChallengeType(Enum):
    GSR = "gsr"  # Current challenge type
    ACTION_SPOTTING = "action_spotting"  # Future challenge type
    
@dataclass
class ChallengeMetadata:
    """Base class for challenge metadata"""
    challenge_id: str
    type: ChallengeType
    created_at: datetime
    
@dataclass
class GSRChallenge(ChallengeMetadata):
    """Game State Reconstruction challenge specifics"""
    video_url: str
    num_frames_to_validate: int = 2  # Default number of frames to validate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert challenge to dictionary for sending to miners"""
        return {
            "challenge_id": str(self.challenge_id),
            "type": self.type.value,
            "video_url": self.video_url,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
@dataclass
class GSRResponse:
    """Expected response format for GSR challenges"""
    challenge_id: str
    frames: Dict[int, Dict[str, Any]]  # frame_number -> frame data
    processing_time: float
    node_id: Optional[int] = None
    miner_hotkey: Optional[str] = None
    response_id: Optional[int] = None
    received_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    evaluated: Optional[bool] = False
    score: Optional[float] = None
    evaluated_at: Optional[datetime] = None
    response_data: Optional[Dict] = None
    
    def get_frame_numbers(self) -> List[int]:
        """Get sorted list of frame numbers"""
        return sorted(self.frames.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for storage"""
        return {
            "challenge_id": self.challenge_id,
            "frames": self.frames,
            "processing_time": self.processing_time,
            "node_id": self.node_id,
            "miner_hotkey": self.miner_hotkey,
            "response_id": self.response_id,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "evaluated": self.evaluated,
            "score": self.score,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "response_data": self.response_data
        }
        
class ValidationResult:
    def __init__(self, 
                 score: float,
                 frame_scores: Dict[int, float],  # Use int frame numbers
                 feedback: str,
                 processing_time: Optional[float] = None,
                 error: Optional[str] = None):
        self.score = score
        self.frame_scores = frame_scores  # Individual scores for each frame
        self.feedback = feedback
        self.processing_time = processing_time
        self.error = error
        
    @property
    def is_valid(self) -> bool:
        return self.error is None

@dataclass
class ChallengeTask:
    """Represents an active challenge task being processed"""
    node_id: int
    task: 'asyncio.Task'
    timestamp: datetime
    challenge: GSRChallenge
    miner_hotkey: str
