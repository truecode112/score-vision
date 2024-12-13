# Subnet 44 - Soccer Video Analysis

This repository contains both the miner and validator components for Subnet 44, focusing on soccer video analysis.

## Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) for dependency management

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd sn44
```

2. Create and activate a virtual environment with Python 3.10:

```bash
uv venv --python=3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

If you don't have Python 3.10 installed, you can install it first:

```bash
uv python install 3.10
```

3. Install components:

For both miner and validator:

```bash
uv pip install -e ".[miner,validator]"
```

Or individually:

For miner only:

```bash
uv pip install -e ".[miner]"
```

For validator only:

```bash
uv pip install -e ".[validator]"
```

## Running the Services

### Miner

```bash
cd miner
uvicorn main:app --reload --port 8000
```

### Validator

```bash
cd validator
uvicorn validator.main:app --reload --port 8001
```

The services will be available at:

- Miner: `http://localhost:8000`
- Validator: `http://localhost:8001`

## Development

Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest miner/tests
pytest validator/tests
```

## License

MIT
