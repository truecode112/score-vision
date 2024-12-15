import asyncio
import json
from datetime import datetime, timezone
import httpx
from fiber.logging_utils import get_logger
from fiber.validator import client as validator
from fiber import Keypair
from validator.challenge.challenge_types import GSRChallenge, GSRResponse
from validator.config import CHALLENGE_TIMEOUT

logger = get_logger(__name__)

async def send_challenge(
    challenge: GSRChallenge,
    server_address: str,
    hotkey: str,
    keypair: Keypair,
    node_id: int,
    db_manager=None,
    client: httpx.AsyncClient = None,
    timeout: float = CHALLENGE_TIMEOUT.total_seconds()  # Use config timeout in seconds
) -> httpx.Response:
    """Send a challenge to a miner node using fiber 2.0.0 protocol."""
    endpoint = "/soccer/challenge"
    payload = challenge.to_dict()

    logger.info(f"Preparing to send challenge to node {node_id}")
    logger.info(f"  Server address: {server_address}")
    logger.info(f"  Endpoint: {endpoint}")
    logger.info(f"  Hotkey: {hotkey}")
    logger.info(f"  Challenge ID: {challenge.challenge_id}")
    logger.info(f"  Payload: {json.dumps(payload, indent=2)}")

    try:
        # First, store the challenge in the challenges table
        if db_manager:
            logger.debug(f"Storing challenge {challenge.challenge_id} in database")
            db_manager.store_challenge(
                challenge_id=challenge.challenge_id,
                challenge_type=str(challenge.type),  # Convert enum to string
                video_url=challenge.video_url,
                task_name="soccer"
            )

        # Record the assignment
        if db_manager:
            logger.debug(f"Recording challenge assignment in database")
            db_manager.assign_challenge(challenge.challenge_id, hotkey, node_id)

        # Create client if not provided
        should_close_client = False
        if client is None:
            logger.debug("Creating new HTTP client")
            client = httpx.AsyncClient(timeout=timeout)
            should_close_client = True

        try:
            # Mark as sent BEFORE awaiting response
            sent_time = datetime.now(timezone.utc)
            if db_manager:
                logger.debug("Marking challenge as sent in database")
                db_manager.mark_challenge_sent(challenge.challenge_id, hotkey)

            logger.debug("Sending challenge request...")
            
            # Send the challenge using fiber validator client with long timeout
            response = await validator.make_non_streamed_post(
                httpx_client=client,
                server_address=server_address,
                validator_ss58_address=keypair.ss58_address,
                miner_ss58_address=hotkey,
                keypair=keypair,
                endpoint=endpoint,
                payload=payload,
                timeout=timeout
            )
            
            received_time = datetime.now(timezone.utc)
            processing_time = (received_time - sent_time).total_seconds()
            
            logger.debug(f"Got response with status code: {response.status_code}")
            
            try:
                response.raise_for_status()
                response_data = response.json()
                
                # Convert frames list to dictionary keyed by frame number
                frames_data = {}
                if 'frames' in response_data and isinstance(response_data['frames'], list):
                    for frame in response_data['frames']:
                        if 'frame_number' in frame:
                            frame_num = str(frame['frame_number'])  # Convert to string key
                            frames_data[frame_num] = frame
                else:
                    logger.warning("No frames data found in response")
                
                # Create GSRResponse with parsed data
                gsr_response = GSRResponse(
                    challenge_id=challenge.challenge_id,
                    frames=frames_data,
                    processing_time=response_data.get('processing_time', 0.0),
                    node_id=node_id,
                    miner_hotkey=hotkey,
                    received_at=received_time
                )
                
                # Store response in responses table
                if db_manager:
                    logger.debug("Storing response in database")
                    response_id = db_manager.store_response(
                        challenge_id=challenge.challenge_id,
                        miner_hotkey=hotkey,
                        response=gsr_response,
                        node_id=node_id,
                        processing_time=processing_time
                    )
                    
                    logger.info(f"Stored response {response_id} in database")
                
            except Exception as e:
                logger.error(f"Response error: {str(e)}")
                logger.error(f"Response body: {response.text}")
                raise

            logger.info(f"Challenge {challenge.challenge_id} sent successfully to {hotkey} (node {node_id})")
            return response
            
        finally:
            if should_close_client:
                logger.debug("Closing HTTP client")
                await client.aclose()
            
    except Exception as e:
        if db_manager:
            logger.debug("Marking challenge as failed in database")
            db_manager.mark_challenge_failed(challenge.challenge_id, hotkey)
        logger.error(f"Failed to send challenge {challenge.challenge_id} to {hotkey} (node {node_id}): {str(e)}")
        logger.exception("Full error traceback:")
        raise
