import os
import json
import cv2
from pathlib import Path
from validator.evaluation.evaluation import GSRValidator
from validator.challenge.challenge_types import GSRResponse

async def test_evaluation():
    """Test evaluation using sample files from debug_frames directory."""
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize validator
    validator = GSRValidator(openai_api_key=openai_api_key)

    # Load test files
    debug_frames_dir = Path("debug_frames")
    response_file = debug_frames_dir / "challenge-response.json"
    frame_file = debug_frames_dir / "challenge_7_frame_352_miner_5H3MgtYa85LhpPt8xee2KDpuZwzD4GG4VhFsxmxXFvviKLi2.jpg"

    # Load response data
    with open(response_file, 'r') as f:
        response_data = json.load(f)

    # Create GSRResponse object
    response = GSRResponse(
        challenge_id=response_data.get('challenge_id', '7'),
        miner_hotkey=response_data.get('miner_hotkey', '5H3MgtYa85LhpPt8xee2KDpuZwzD4GG4VhFsxmxXFvviKLi2'),
        frames=response_data.get('frames', {}),
        node_id=response_data.get('node_id', 1)
    )

    # Load frame image
    frame = cv2.imread(str(frame_file))
    if frame is None:
        raise ValueError(f"Failed to load image: {frame_file}")

    # Get reference counts
    reference_counts = validator.get_reference_counts(frame)
    print("\nReference Counts:")
    print(json.dumps(reference_counts, indent=2))

    # Evaluate frame
    frame_evaluation = validator.evaluate_frame(frame, response.frames.get('352', {}), reference_counts)
    print("\nFrame Evaluation:")
    print(json.dumps(frame_evaluation, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_evaluation())
