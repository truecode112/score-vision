# SN44 Validator

This is the validator component for Subnet 44 (Soccer Video Analysis). For full subnet documentation, please see the [main README](../README.md).

## Quick Start

1. Follow the installation steps in the main README first.

2. Configure the validator:

```bash
# Copy example environment file
cp validator/.env.example validator/.env

# Edit .env with your settings
nano validator/.env
```

Required configurations:

- `OPENAI_API_KEY`: Your GPT-4o capable API key
- `WALLET_NAME`: Your validator wallet name
- `HOTKEY_NAME`: Your validator hotkey name

## Operational Overview

The validator runs several concurrent loops to manage the subnet:

### 1. Challenge Management Loop (Every 5 minutes)

- Fetches new soccer video challenges from the challenge API
- Identifies active miners with sufficient stake (<100 TAO)
- Checks miner availability via health endpoints
- Distributes challenges to available miners
- Tracks challenge distribution and timeouts

### 2. Response Collection Loop

- Monitors for incoming challenge responses
- Validates response format and completeness
- Stores responses in the database for evaluation
- Handles timeouts and failed responses
- Manages concurrent response processing

### 3. Evaluation Loop (Continuous)

Note – this is in ongoing development and likely to change significantly.

- Processes stored responses in batches
- Uses random frames from the challenges
- Uses GPT-4o to validate frame annotations:
  - Player detection accuracy
  - Goalkeeper identification
  - Referee detection
  - Ball tracking
- Calculates frame-level scores
- Stores evaluation results

### 4. Weight Setting Loop (Every 21 minutes)

Note – this is in ongoing development and likely to change significantly.

- Aggregates recent evaluation scores
- Calculates miner performance metrics:
  - Evaluation accuracy (60%)
  - Availability (30%)
  - Response speed (10%)
- Updates miner weights on-chain
- Manages reward distribution

## Running the Validator

### Using PM2 (Recommended)

```bash
cd validator
pm2 start \
  --name "sn44-validator" \
  --interpreter "../.venv/bin/python" \
  "../.venv/bin/uvicorn" \
  -- main:app --port 8000
```

Common PM2 commands:

```bash
pm2 logs sn44-validator    # View logs
pm2 stop sn44-validator    # Stop validator
pm2 restart sn44-validator # Restart validator
```

### Manual Run

```bash
cd validator
uvicorn main:app --reload --port 8000
```

## Configuration Reference

Key environment variables in `.env`:

```bash
# Network
NETUID=261                                    # Subnet ID (261 for testnet)
SUBTENSOR_NETWORK=test                        # Network type (test/local)
SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443  # Network address

# Validator
VALIDATOR_PORT=8000                           # API port
VALIDATOR_HOST=0.0.0.0                        # API host
MIN_STAKE_THRESHOLD=2                         # Minimum stake requirement

# API Keys
OPENAI_API_KEY=your_key_here                  # GPT-4o capable key
```

For advanced configuration options and architecture details, see the [main README](../README.md).
