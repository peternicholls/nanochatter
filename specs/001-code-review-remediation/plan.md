# Implementation Plan: Code Review Remediation Baseline

**Branch**: `[001-code-review-remediation]` | **Date**: 2026-03-21 | **Spec**: `/Users/peternicholls/Dev/nanochatter/specs/001-code-review-remediation/spec.md`
**Input**: Feature specification from `/specs/001-code-review-remediation/spec.md`

**Note**: This plan was produced for the existing nanochat repository on the active remediation branch.

## Summary

Restore contributor trust in the current branch by making the package importable through one supported environment workflow, standardizing one canonical automated test command, fixing the known failing contracts in runtime telemetry, native helper build freshness, chat role validation, and cache precision selection, and reducing runtime surprise from undeclared dependencies and eager training-only imports. The implementation will stay scoped to the audited defects, prefer existing repo structure over refactors, and document one contributor-facing workflow that matches automation.

## Technical Context

**Language/Version**: Python >=3.10 for the main project; Swift package under `swift/NanochatMLXStub` for the native helper  
**Primary Dependencies**: PyTorch, FastAPI, Uvicorn, HuggingFace Datasets, Tokenizers, Tiktoken, Transformers, optional MLX on macOS, plus direct runtime imports that need explicit declaration (`filelock`, `requests`, `pyarrow`, `PyYAML`, `Jinja2`)  
**Storage**: File-based project artifacts and checkpoints under the repo and nanochat base directory; no database  
**Testing**: `pytest` for Python test suite; canonical planned invocation is `python -m pytest -q` from the supported project environment  
**Target Platform**: Linux/CUDA, Linux CPU-only, and macOS/Apple Silicon with MPS and native Swift helper support  
**Project Type**: Single Python package with CLI and web entrypoints, plus a native Swift helper package  
**Performance Goals**: Preserve current inference/training behavior while removing correctness mismatches; avoid regressions in native helper freshness decisions and cache precision behavior  
**Constraints**: Keep remediation branch-local, avoid unrelated architectural refactors, preserve existing public entrypoints, and keep chat role scope limited to `user` and `assistant`  
**Scale/Scope**: Focused remediation across `pyproject.toml`, `nanochat/`, `scripts/`, `tests/`, and `swift/` integration seams identified in the 2026-03-21 review

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

No repository-specific constitution file exists under `.specify/memory/constitution.md`; the fallback governance source in this repo is the unfilled template, so no additional project-specific constitutional rules are currently ratified.

Fallback gates applied for this feature:

- Scope gate: PASS. The feature is restricted to defects explicitly called out in the 2026-03-21 code review.
- Simplicity gate: PASS. The plan uses existing package, script, and test structure rather than introducing new subsystems.
- Testability gate: PASS. Every user story has independent verification, and the plan centers the canonical regression command plus targeted contract checks.
- Branch-local workflow gate: PASS. All planning artifacts live under `specs/001-code-review-remediation/` on the active feature branch.

Post-design re-check:

- Scope gate: PASS. Design artifacts keep the work bounded to packaging/test workflow, reviewed contract mismatches, and dependency/runtime cleanup.
- Simplicity gate: PASS. No new services, packages, or protocol surfaces were added beyond documentation artifacts.
- Testability gate: PASS. Research, contracts, and quickstart preserve concrete validation steps for each remediation area.
- Branch-local workflow gate: PASS. All generated artifacts remain inside the active feature directory.

## Project Structure

### Documentation (this feature)

```text
specs/001-code-review-remediation/
├── plan.md
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

