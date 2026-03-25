# Quickstart: Code Review Remediation Baseline

## Goal

Validate the remediation plan in the existing nanochat codebase once implementation is complete.

## 1. Prepare the environment

1. From repo root, create or sync the supported environment for the current platform.
2. Use the platform-appropriate dependency sync path already documented by the repo:
   - `uv sync --extra gpu`
   - `uv sync --extra cpu`
   - `uv sync --extra macos`
3. Confirm the package is importable through the same workflow:

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
   - Run the memory-helper tests that cover absent or partially stubbed accelerator backends.
3. Native helper build contract:
   - Run the tests covering native helper build freshness and output-path assumptions.
4. Chat role contract:
   - Run the validation coverage that confirms `system` is rejected and only `user`/`assistant` are accepted.
5. Runtime precision policy:
   - Run inference tests that confirm cache allocation follows the configured compute precision.
6. Runtime import boundary:
   - Exercise runtime and evaluation import/load paths without training-only tokenization tooling and confirm the supported paths still work.
7. Reporting diagnostics:
   - Exercise report helper error handling and confirm degraded behavior is diagnosable.

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