from pydantic import BaseModel, Field
from typing import Optional

class Config(BaseModel):
    """Configuration for the miner service."""
    
    # Device configuration
    device: str = Field(
        default="cpu",
        description="Device to use for model inference ('cuda', 'mps', or 'cpu')"
    )
    
    # Video processing timeouts (in seconds)
    cuda_timeout: float = Field(
        default=900.0,  # 15 minutes for CUDA
        description="Maximum processing time for CUDA device"
    )
    mps_timeout: float = Field(
        default=1800.0,  # 30 minutes for MPS
        description="Maximum processing time for MPS device"
    )
    cpu_timeout: float = Field(
        default=10800.0,  # 3 hours for CPU
        description="Maximum processing time for CPU device"
    )
    
    # Model paths (relative to data directory)
    player_model_path: str = Field(
        default="football-player-detection.pt",
        description="Path to player detection model"
    )
    pitch_model_path: str = Field(
        default="football-pitch-detection.pt",
        description="Path to pitch detection model"
    )
    ball_model_path: str = Field(
        default="football-ball-detection.pt",
        description="Path to ball detection model"
    )
    
    class Config:
        env_prefix = "MINER_"