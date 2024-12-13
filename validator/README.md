# Score Subnet Validator

The Score Subnet Validator is a Bittensor subnet validator for evaluating Game State Reconstruction (GSR) challenges in soccer videos. It manages the full lifecycle of challenges, from distribution to evaluation and reward calculation.

## Testnet

Testnet netuid: 261

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) for dependency management
- [PM2](https://pm2.keymetrics.io/) for process management (optional)
- OpenAI API key (for GPT-4V)
- Substrate account keypair

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd sn44
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
uv pip install -e ".[validator]"
```

## Running the Validator

### Option 1: Direct Python Execution

```bash
cd validator
python main.py
```

### Option 2: Using PM2 (Recommended for Production)

1. Install PM2 if you haven't already:

```bash
npm install -g pm2
```

2. Start the validator:

```bash
cd validator
pm2 start ecosystem.config.js
```

PM2 Commands:

```bash
pm2 list                    # List all processes
pm2 logs sn44-validator    # View logs
pm2 stop sn44-validator    # Stop the validator
pm2 restart sn44-validator # Restart the validator
pm2 delete sn44-validator  # Remove from PM2
```

## Configuration

1. Set up environment variables in `dev.env`:

```bash
OPENAI_API_KEY=your_api_key_here
SUBSTRATE_URL=your_substrate_url
VALIDATOR_SEED=your_validator_seed
DB_CONNECTION=your_db_connection_string
```

2. Configure system parameters in config.py:

- `CHALLENGE_INTERVAL`: Time between challenge distributions
- `WEIGHTS_INTERVAL`: Time between weight updates
- Database settings
- Network parameters

## System Architecture

The validator operates through several coordinated components:

### 1. Challenge Management

- Generates and distributes GSR challenges to miner nodes
- Manages challenge scheduling and distribution (every 2 hours)
- Tracks challenge assignments and completions in database
- Encrypts challenge data for secure transmission

### 2. Response Collection

- Continuously monitors for challenge responses from miners
- Collects and validates response data
- Stores responses in database for evaluation
- Handles timeouts and failed responses

### 3. Evaluation System

- Evaluates GSR challenge responses using Vision Language Models (VLM)
- Performs batch frame analysis for efficiency
- Validates object detection quality:
  - Players (field players)
  - Goalkeepers
  - Referees
  - Soccer ball
- Provides detailed scoring and feedback

### 4. Weight Management

- Calculates and updates miner weights based on performance
- Periodically updates weights on-chain
- Implements scoring algorithms for fair reward distribution

## Development

To run tests:

```bash
pytest
```

## License

MIT
