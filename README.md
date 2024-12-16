# Subnet 44 - Soccer Video Analysis

This repository contains both the miner and validator components for Subnet 44, focusing on soccer video analysis.

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) for dependency management

## Installation

### 0. Preliminary Setup

1. Bootstrap

Get code and run bootstrap.sh script

```bash
# Clone repository
git clone https://github.com/score-protocol/sn44.git
cd sn44
sudo chmod +x bootstrap.sh
./bootstrap.sh
```

This will download the subnet code and install the core prerequestits

3. Copy relevant hotkey and public coldkey from local to remote:
   Or import the whole wallets on remote. Whichvever you prefer.

```bash
# Create hotkey directory
mkdir -p /root/.bittensor/wallets/[walletname]/hotkeys/

# Copy hotkey from local to remote
scp ~/.bittensor/wallets/[walletname]/hotkeys/[hotkeyname] [user]@[SERVERIP]:/root/.bittensor/wallets/[walletname]/hotkeys/[hotkeyname]

# Copy coldkey public key from lcoal to remote
scp ~/.bittensor/wallets/[walletname]/coldkeypub.txt [user]@[SERVERIP]:/root/.bittensor/wallets/[walletname]/coldkeypub.txt


```

# On macOS and Linux.

### 1. Setup

```bash
# Create and activate virtual environment
uv venv --python=3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Components

For Miner:

```bash
# Install miner dependencies
uv pip install -e ".[miner]"
cp miner/.env.example miner/.env
```

For Validator:

```bash
uv pip install -e ".[validator]"
cp validator/.env.example validator/dev.env
```

## Registering IP to chain

This is an important step for Miners to complete – it manually sets the IP and port for their node on chain, and this is what the Validators will use to send the challenges to.

1. Find your IP on remote server

```bash
curl ifconfig.me
```

This requires coldkey to be local – so run it where you deem fit.

For running locally, first install Fiber.

```bash
pip install "git+https://github.com/rayonlabs/fiber.git@2.0.1#egg=fiber[full]"
```

Then run the fiber-post-ip script, naming your miners IP, port, wallet name and hotkey.

```bash
fiber-post-ip --netuid 261 --subtensor.network test --external_port 7999 --wallet.name default --wallet.hotkey default --external_ip [YOUR-IP]
```

fiber-post-ip --netuid 261 --subtensor.network test --external_port 7999 --wallet.name 1valid --wallet.hotkey miner1 --external_ip 178.62.9.109

## Running the Services

### Miner

#### Testing the pipeline

You can use this to test inference

```bash
cd miner
python scripts/test_pipeline.py
```

#### pm2 (preferered)

```bash
cd miner
pm2 start \
  --name "sn44-miner" \
  --interpreter "../.venv/bin/python" \
  "../.venv/bin/uvicorn" \
  -- main:app --host 0.0.0.0 --port 7999
```

#### Manually

```bash
cd miner
uvicorn main:app --reload --host 0.0.0.0 --port 7999
```

### Validator

#### pm2 (preferred)

```bash
cd validator
pm2 start \
  --name "sn44-validator" \
  --interpreter "../.venv/bin/python" \
  "../.venv/bin/uvicorn" \
  -- main:app --port 8000

```

#### Manually

```bash
cd validator
uvicorn validator.main:app --reload --port 8000
```

## License

MIT
