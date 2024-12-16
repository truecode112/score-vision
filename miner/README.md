# SN44 Miner

This is the miner component for Subnet 44 (Soccer Video Analysis). For full subnet documentation, please see the [main README](../README.md).

## Quick Start

1. Follow the installation steps in the main README first.

2. Configure the miner:

```bash
# Copy example environment file
cp miner/.env.example miner/.env

# Edit .env with your settings
nano miner/.env
```

Required configurations:

- `WALLET_NAME`: Your miner wallet name
- `HOTKEY_NAME`: Your miner hotkey name
- `DEVICE`: Computing device ('cuda', 'cpu', or 'mps')

3. Download required models:

```bash
cd miner
python scripts/download_models.py
```

## Operational Overview

The miner operates several key processes to handle soccer video analysis:

### 1. Challenge Reception

- Listens for incoming challenges from validators
- Validates challenge authenticity using cryptographic signatures
- Downloads video content from provided URLs
- Manages concurrent challenge processing
- Implements exponential backoff for failed downloads

### 2. Video Processing Pipeline

- Loads video frames efficiently using OpenCV
- Processes frames through multiple detection models:
  - Player detection and tracking
  - Goalkeeper identification
  - Referee detection
  - Ball tracking
- Manages GPU memory for optimal performance
- Implements frame batching for efficiency

### 3. Response Generation

- Generates standardized bounding box annotations
- Formats responses according to subnet protocol
- Includes confidence scores for detections
- Implements quality checks before submission
- Handles response encryption and signing

### 4. Health Management

- Maintains availability endpoint for validator checks
- Monitors system resources (GPU/CPU usage)
- Implements graceful challenge rejection when overloaded
- Tracks processing metrics and timings
- Manages concurrent request limits

## Running the Miner

### Using PM2 (Recommended)

```bash
cd miner
pm2 start \
  --name "sn44-miner" \
  --interpreter "../.venv/bin/python" \
  "../.venv/bin/uvicorn" \
  -- main:app --host 0.0.0.0 --port 7999
```

Common PM2 commands:

```bash
pm2 logs sn44-miner     # View logs
pm2 stop sn44-miner     # Stop miner
pm2 restart sn44-miner  # Restart miner
```

### Manual Run

```bash
cd miner
uvicorn main:app --reload --host 0.0.0.0 --port 7999
```

### Testing the Pipeline

To test the inference pipeline locally:

```bash
cd miner
python scripts/test_pipeline.py
```

## Configuration Reference

Key environment variables in `.env`:

```bash
# Network
NETUID=261                                    # Subnet ID (261 for testnet)
SUBTENSOR_NETWORK=test                        # Network type (test/local)
SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443  # Network address

# Miner
WALLET_NAME=default                           # Your wallet name
HOTKEY_NAME=default                           # Your hotkey name
MIN_STAKE_THRESHOLD=2                         # Minimum stake requirement

# Hardware
DEVICE=cuda                                   # Computing device (cuda/cpu/mps)
```

## Troubleshooting

### Common Issues

1. **Video Download Failures**

   - System implements automatic retries (3 attempts)
   - Uses exponential backoff between attempts
   - Check network connectivity if persistent

2. **Out of Memory**

   - Reduce batch size in configuration
   - Monitor GPU memory usage
   - Consider using CPU if GPU memory limited

3. **Model Loading Errors**
   - Verify models in `miner/data` directory
   - Run download script again if missing
   - Check device compatibility

For advanced configuration options and architecture details, see the [main README](../README.md).
