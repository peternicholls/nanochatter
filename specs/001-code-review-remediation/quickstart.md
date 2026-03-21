# Quickstart: Code Review Remediation Baseline

## Goal

Validate the remediation plan in the existing nanochat codebase once implementation is complete.

## 1. Prepare the environment

1. Create or sync the supported environment for the current platform.
2. Use the platform-appropriate dependency sync path already documented by the repo:
   - `uv sync --extra gpu`
   - `uv sync --extra cpu`
   - `uv sync --extra macos`
3. Ensure the project is installed through the documented package workflow so `nanochat` is importable.

## 2. Run the canonical regression command

1. Activate the project environment if needed.
2. Run:

```bash
python -m pytest -q
```

3. Confirm the suite finishes without remediation-scope failures.

## 3. Run targeted contract checks

1. Packaging/importability:
   - Confirm `nanochat` imports successfully from the environment without manual path overrides.
2. Runtime telemetry:
   - Run the memory-helper tests that cover absent or partially stubbed accelerator backends.
3. Native helper build contract:
   - Run the tests covering native helper build freshness and output-path assumptions.
4. Chat role contract:
   - Run the validation coverage that confirms `system` is rejected and only `user`/`assistant` are accepted.
5. Runtime precision policy:
   - Run inference tests that confirm cache allocation follows the configured compute precision.
6. Runtime import boundary:
   - Exercise a runtime-only import/load path without training-only tokenization tooling and confirm it still works.
7. Reporting diagnostics:
   - Exercise report helper error handling and confirm degraded behavior is diagnosable.

## 4. Review documentation alignment

1. Confirm the README or contributor docs expose only one canonical automated test command.
2. Confirm the supported chat roles in docs and runtime behavior match.
3. Confirm dependency metadata matches direct runtime imports.

## 5. Expected outcome

1. The project is importable from the supported environment.
2. The canonical regression command is green for remediation-scope issues.
3. Reviewed runtime and integration contracts match their documentation.
4. Runtime-only paths no longer depend on training-only tokenization tooling unless explicitly requested.