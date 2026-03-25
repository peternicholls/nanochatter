# Implementation Plan: Code Review Remediation Baseline

**Branch**: `[001-code-review-remediation]` | **Date**: 2026-03-21 | **Spec**: `/Users/peternicholls/Dev/nanochatter/specs/001-code-review-remediation/spec.md`
**Input**: Feature specification from `/specs/001-code-review-remediation/spec.md` plus remediation decisions from `/specs/001-code-review-remediation/remediation.md`

**Note**: This plan was produced for the existing nanochat repository on the active remediation branch.

## Summary

Restore contributor trust in the current branch by making the package importable through one explicit `uv` workflow, standardizing one canonical automated test command shared by contributors and automation, fixing the known failing contracts in runtime telemetry, native helper build freshness, chat role validation, and cache precision selection, and reducing runtime surprise from undeclared dependencies and eager training-only imports. The implementation will stay scoped to the audited defects, prefer existing repo structure over refactors, and document one contributor-facing workflow that matches automation exactly.

## Technical Context

**Language/Version**: Python >=3.10 for the main project; Swift package under `swift/NanochatMLXStub` for the native helper  
**Primary Dependencies**: PyTorch, FastAPI, Uvicorn, HuggingFace Datasets, Tokenizers, Tiktoken, Transformers, optional MLX on macOS, plus direct runtime imports that need explicit declaration (`filelock`, `requests`, `pyarrow`, `PyYAML`, `Jinja2`)  
**Storage**: File-based project artifacts and checkpoints under the repo and nanochat base directory; no database  
**Testing**: `pytest` for Python test suite; canonical planned invocation is `uv run python -m pytest -q` from repo root after `uv sync --extra <platform>`  
**Target Platform**: Linux/CUDA, Linux CPU-only, and macOS/Apple Silicon with MPS and native Swift helper support  
**Project Type**: Single Python package with CLI and web entrypoints, plus a native Swift helper package  
**Performance Goals**: Preserve current inference/training behavior while removing correctness mismatches; avoid regressions in native helper freshness decisions and cache precision behavior  
**Constraints**: Keep remediation branch-local, avoid unrelated architectural refactors, preserve existing public entrypoints, keep chat role scope limited to `user` and `assistant`, and keep contributor and automation workflow parity  
**Scale/Scope**: Focused remediation across `pyproject.toml`, `nanochat/`, `scripts/`, `tests/`, and `swift/` integration seams identified in the 2026-03-21 review

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The repository constitution now exists at `.specify/memory/constitution.md` and applies directly to this plan.

Pre-design gates:

- Correctness-before-expansion gate: PASS. The feature remains restricted to defects and workflow trust issues called out in the 2026-03-21 review.
- Canonical-workflow gate: PASS. The plan standardizes `uv sync --extra <platform>` plus `uv run python -m pytest -q` as the shared contributor and automation path.
- Contract-backed-change gate: PASS. Each reviewed runtime or integration contract maps to targeted regression coverage.
- Runtime-boundary gate: PASS. The plan includes explicit dependency, import-boundary, and reporting-diagnostic work.
- Documentation-portability gate: PASS. The plan includes workflow/doc alignment and portability cleanup rather than relying on stale guidance.

Post-design re-check:

- Correctness-before-expansion gate: PASS. Design artifacts stay bounded to packaging/test workflow, reviewed contract mismatches, dependency hygiene, and documentation truthfulness.
- Canonical-workflow gate: PASS. Quickstart, contracts, and tasks all use the same `uv` workflow.
- Contract-backed-change gate: PASS. Targeted tests cover telemetry, native helper freshness, role validation, precision policy, dependency boundaries, and reporting diagnostics.
- Runtime-boundary gate: PASS. Evaluation-path dependency coverage is now part of the planned verification.
- Documentation-portability gate: PASS. The feature includes doc reconciliation and automation-smoke alignment work.

## Project Structure

### Documentation (this feature)

```text
specs/001-code-review-remediation/
├── plan.md
├── remediation.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── chat-role-validation.md
│   └── contributor-test-workflow.md
└── tasks.md
```

### Source Code (repository root)

```text
nanochat/
├── __init__.py
├── checkpoint_manager.py
├── common.py
├── engine.py
├── report.py
├── swift_build.py
├── swift_stub_engine.py
└── tokenizer.py

scripts/
├── base_eval.py
├── chat_web.py
└── mlx_swift_stub.py

tests/
├── test_common_mps_memory.py
├── test_engine.py
└── test_swift_stub_engine.py

swift/
├── Build/
└── NanochatMLXStub/

pyproject.toml
README.md
```

**Structure Decision**: Keep all implementation work inside the existing single-project repository layout. The remediation spans the Python package in `nanochat/`, entry scripts in `scripts/`, regression coverage in `tests/`, the native helper package under `swift/`, and packaging metadata in `pyproject.toml` and documentation.

## Complexity Tracking

No constitution violations or complexity exceptions are required for this plan.

