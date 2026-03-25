# Quickstart: Code Review Remediation Baseline

## Goal

Validate the remediation plan in the existing nanochat codebase once implementation is complete.

## 1. Prepare the environment

1. From repo root, create or sync the supported environment for the current platform.
2. Use the platform-appropriate dependency sync path already documented by the repo:
   - `uv sync --extra gpu`
   - `uv sync --extra cpu`
   - `uv sync --extra macos`
3. Include the test dependency group when validating locally if your environment does not already include it: `uv sync --extra <platform> --group dev`
4. Confirm the package is importable through the same workflow:

```bash
uv run python -c "import nanochat"
```

## 2. Run the canonical regression command

1. Run:

```bash
uv run python -m pytest -q
```

3. Confirm the suite finishes without remediation-scope failures.

## 3. Run targeted contract checks

1. Packaging/importability:
   - Confirm `uv run python -c "import nanochat"` succeeds without manual path overrides.
2. Runtime telemetry:
   - Run `uv run python -m pytest -q tests/test_common_mps_memory.py`.
3. Native helper build contract:
   - Run `uv run python -m pytest -q tests/test_swift_stub_engine.py`.
4. Chat role contract:
   - Run `uv run python -m pytest -q tests/test_chat_web_validation.py`.
5. Runtime precision policy:
   - Run `uv run python -m pytest -q tests/test_engine.py`.
6. Runtime import boundary:
   - Run `uv run python -m pytest -q tests/test_runtime_boundaries.py`.
7. Reporting diagnostics:
   - Run `uv run python -m pytest -q tests/test_report.py`.

## 4. Review documentation alignment

1. Confirm the README or contributor docs expose only one canonical automated test command.
2. Confirm the automation smoke workflow uses the same canonical setup and regression commands.
3. Confirm the supported chat roles in docs and runtime behavior match.
4. Confirm dependency metadata matches direct runtime and evaluation imports.

## 5. Expected outcome

1. The project is importable from the supported environment.
2. The canonical regression command is green for remediation-scope issues.
3. Reviewed runtime and integration contracts match their documentation.
4. Runtime-only paths no longer depend on training-only tokenization tooling unless explicitly requested.