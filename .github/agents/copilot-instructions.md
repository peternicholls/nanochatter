# nanochatter Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-21

## Active Technologies

- Python >=3.10 for the main project; Swift package under `swift/NanochatMLXStub` for the native helper + PyTorch, FastAPI, Uvicorn, HuggingFace Datasets, Tokenizers, Tiktoken, Transformers, optional MLX on macOS, plus direct runtime imports that need explicit declaration (`filelock`, `requests`, `pyarrow`, `PyYAML`, `Jinja2`) (001-code-review-remediation)

## Project Structure

```text
backend/
frontend/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python >=3.10 for the main project; Swift package under `swift/NanochatMLXStub` for the native helper: Follow standard conventions

## Recent Changes

- 001-code-review-remediation: Added Python >=3.10 for the main project; Swift package under `swift/NanochatMLXStub` for the native helper + PyTorch, FastAPI, Uvicorn, HuggingFace Datasets, Tokenizers, Tiktoken, Transformers, optional MLX on macOS, plus direct runtime imports that need explicit declaration (`filelock`, `requests`, `pyarrow`, `PyYAML`, `Jinja2`)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
