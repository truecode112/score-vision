#!/usr/bin/env python3
import os
import json
import cv2
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from validator.evaluation.prompts import COUNT_PROMPT, VALIDATION_PROMPT

def encode_image(image_path: str) -> str:
    """Encode image as base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def test_prompts():
    """Test the COUNT_PROMPT and VALIDATION_PROMPT directly with the VLM."""
    # Load environment variables from dev.env in current directory
    current_dir = Path(__file__).parent
    dev_env_path = current_dir / "dev.env"
    
    if not dev_env_path.exists():
        raise FileNotFoundError(f"dev.env not found at {dev_env_path}")
    
    print(f"Loading environment from: {dev_env_path}")
    load_dotenv(dev_env_path)
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set in dev.env")

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Load test image from debug_frames in parent directory
    debug_frames_dir = current_dir / "debug_frames"
    frame_file = debug_frames_dir / "challenge_7_frame_352_miner_5H3MgtYa85LhpPt8xee2KDpuZwzD4GG4VhFsxmxXFvviKLi2.jpg"

    # Check if file exists
    if not frame_file.exists():
        raise FileNotFoundError(f"Frame file not found: {frame_file}")

    print(f"Using frame file: {frame_file}")

    # Encode image
    encoded_image = encode_image(str(frame_file))
    
    # Test COUNT_PROMPT
    print("\nTesting COUNT_PROMPT...")
    count_messages = [
        {
            "role": "system",
            "content": "You are an expert at counting objects in soccer match frames."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": COUNT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", "detail": "high"}}
            ]
        }
    ]

    try:
        count_response = client.chat.completions.create(
            model="gpt-4o",
            messages=count_messages,
            max_tokens=500,
            temperature=0.2
        )
        
        print("Count Response:")
        print(count_response.choices[0].message.content)
        
        # Try to parse the response as JSON
        try:
            count_json = json.loads(count_response.choices[0].message.content)
            print("\nParsed Count JSON:")
            print(json.dumps(count_json, indent=2))
        except json.JSONDecodeError as e:
            print(f"\nFailed to parse count response as JSON: {str(e)}")
            
        # Test VALIDATION_PROMPT with the counts we got
        print("\nTesting VALIDATION_PROMPT...")
        validation_prompt = VALIDATION_PROMPT.format(
            count_json.get('player', 0),
            count_json.get('goalkeeper', 0),
            count_json.get('referee', 0),
            count_json.get('soccer ball', 0)
        )
        
        # Load annotated image
        validation_messages = [
            {
                "role": "system",
                "content": "You are an expert at evaluating bounding box annotations in soccer match frames."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": validation_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}", "detail": "high"}}
                ]
            }
        ]

        validation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=validation_messages,
            max_tokens=1000,
            temperature=0.2
        )
        
        print("Validation Response:")
        print(validation_response.choices[0].message.content)
        
        # Try to parse the response as JSON
        try:
            validation_json = json.loads(validation_response.choices[0].message.content)
            print("\nParsed Validation JSON:")
            print(json.dumps(validation_json, indent=2))
        except json.JSONDecodeError as e:
            print(f"\nFailed to parse validation response as JSON: {str(e)}")

    except Exception as e:
        print(f"Error during API call: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_prompts()) 