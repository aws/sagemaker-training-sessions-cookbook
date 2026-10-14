# Contributing

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install 'sagemaker-train>=1.25.0'  # the training-session SDK
pip install -e '.[dev]'
```

## Running tests

```bash
python -m pytest tests/
```

The suite needs no credentials or SDK session: service clients are mocked and
tokenization is checked on a toy round-trippable tokenizer. A handful of
renderer-fidelity tests compare against real model chat templates and download
tokenizers from Hugging Face on first run; offline they skip with a clear
reason rather than fail. Please add tests alongside behavior changes.

## Code style

- Python 3.11+, type hints on public functions.
- Keep modules focused; shared plumbing belongs in
  `training_sessions_cookbook/service/`.
- Docstrings explain *why*, not just what.

## Pull requests

1. Branch from `main`.
2. Make the change with tests.
3. Ensure `python -m pytest tests/` passes.
4. Open a PR with a clear description of the change and its motivation.
