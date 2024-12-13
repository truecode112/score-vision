# Score Subnet Miner

A Bittensor subnet miner for Game State Reconstruction (GSR) in soccer videos.

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) for dependency management
- [PM2](https://pm2.keymetrics.io/) for process management (optional)

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
uv pip install -e ".[miner]"
```

4. Download required models:

```bash
cd miner
python scripts/download_models.py
```

## Running the Miner

### Option 1: Direct Python with uvicorn (Development)

```bash
cd miner
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Using PM2 (Recommended for Production)

1. Install PM2 if you haven't already:

```bash
npm install -g pm2
```

2. Start the miner:

```bash
cd miner
pm2 start ecosystem.config.js
```

PM2 Commands:

```bash
pm2 list                 # List all processes
pm2 logs sn44-miner     # View logs
pm2 stop sn44-miner     # Stop the miner
pm2 restart sn44-miner  # Restart the miner
pm2 delete sn44-miner   # Remove from PM2
```

## Configuration

1. Set up environment variables in `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
SUBSTRATE_URL=your_substrate_url
MINER_SEED=your_miner_seed
```

## Troubleshooting

### Common Issues

1. **Video Download Timeout**

   - The miner will automatically retry downloads up to 3 times
   - Each retry uses exponential backoff
   - Check your internet connection if issues persist

2. **Missing Models**

   - Run `python scripts/download_models.py` to download required models
   - Models are stored in the `miner/data` directory
   - Check model URLs in `scripts/download_models.py` if downloads fail

3. **Port Already in Use**
   - Change the port using `--port` flag with uvicorn
   - Or modify the port in `ecosystem.config.js` for PM2

## Development

To run tests:

```bash
pytest
```

## License

MIT
