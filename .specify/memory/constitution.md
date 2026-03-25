<!--
Sync Impact Report
- Version change: none -> 1.0.0
- Modified principles:
  - Added I. Correctness Before Expansion
  - Added II. One Canonical Contributor Workflow
  - Added III. Contract-Backed Changes
  - Added IV. Explicit Runtime Boundaries
  - Added V. Evidence-Driven Documentation and Portability
- Added sections:
  - Repository Constraints
  - Development Workflow and Quality Gates
- Removed sections:
  - None
- Templates requiring updates:
  - ✅ reviewed / no text change required: .specify/templates/plan-template.md
  - ✅ reviewed / no text change required: .specify/templates/spec-template.md
  - ✅ reviewed / no text change required: .specify/templates/tasks-template.md
- Feature artifacts updated for alignment:
  - ✅ specs/001-code-review-remediation/spec.md
  - ✅ specs/001-code-review-remediation/plan.md
  - ✅ specs/001-code-review-remediation/tasks.md
  - ✅ specs/001-code-review-remediation/research.md
  - ✅ specs/001-code-review-remediation/data-model.md
  - ✅ specs/001-code-review-remediation/quickstart.md
  - ✅ specs/001-code-review-remediation/contracts/contributor-test-workflow.md
  - ✅ specs/001-code-review-remediation/remediation.md
- Follow-up TODOs:
  - None
-->

# nanochat Constitution

## Core Principles

### I. Correctness Before Expansion

Changes that repair broken behavior, broken workflows, or broken contracts MUST take
priority over new scope, cleanup-only work, or optimization work. Performance wins,
Apple-native experiments, and Swift integration improvements MUST NOT ship behind a
known-red regression signal or a contradictory public contract. When a defect is
found at a boundary between runtime behavior, tests, and docs, the fix MUST restore
one authoritative behavior instead of adding another parallel path.

### II. One Canonical Contributor Workflow

The repository MUST document one canonical setup and test workflow per supported
platform tier and MUST use the same workflow in docs, quickstarts, and automation.
For this repo, the canonical environment manager is `uv`, the supported dependency
sync step is `uv sync --extra <platform>`, and the canonical automated regression
command is `uv run python -m pytest -q`. Alternate commands may exist for local
convenience, but they MUST be treated as secondary and MUST NOT become the only
documented or automated path.

### III. Contract-Backed Changes

Any change to a contributor-facing, runtime, evaluation, or integration contract MUST
be backed by focused regression coverage in the same workstream. Supported behavior,
unsupported behavior, and failure semantics MUST be explicit. A green result is only
meaningful when the tests exercise the contract that the docs and entrypoints claim
to provide.

### IV. Explicit Runtime Boundaries

Runtime and evaluation paths MUST declare their direct dependencies explicitly and
MUST NOT depend on training-only tooling unless the invoked operation explicitly
requires it. Import-time side effects that affect logging, hardware selection, build
state, or execution policy MUST be minimized, predictable, and testable. If a path
depends on a platform-specific capability such as CUDA, MPS, MLX, or Swift build
artifacts, the boundary and fallback behavior MUST be stated directly in code and
docs.

### V. Evidence-Driven Documentation and Portability

Current docs MUST reflect the repository state that contributors can actually run,
while archived docs MUST remain clearly historical. Planning docs, handoff notes,
and operational guides MUST use repo-relative paths where possible, avoid
machine-specific absolute paths, and identify platform-specific limitations without
overstating support. Documentation intended to guide implementation or validation
MUST be updated when the underlying contract changes.

## Repository Constraints

- The repository remains a single Python package with CLI and web entrypoints plus a
  Swift helper package under `swift/NanochatMLXStub`.
- Remediation work MUST stay scoped to concrete defects, workflow trust, contract
  alignment, and documentation accuracy before broader architectural change.
- Cross-platform behavior matters: Linux/CUDA, Linux CPU-only, and macOS/Apple
  Silicon portability paths are all first-class planning concerns even when a change
  is developed on one machine.
- Apple-native work MAY advance performance or ergonomics, but portability and
  fallback behavior MUST remain explicit rather than implied.

## Development Workflow and Quality Gates

- Every spec, plan, and tasks set MUST identify the canonical contributor workflow it
  relies on.
- Plans MUST pass a constitution check before implementation starts and MUST explain
  any justified complexity exceptions explicitly.
- Tasks for contract changes MUST include the regression coverage needed to prove the
  contract.
- Documentation changes that alter contributor workflow, platform support, or runtime
  contracts MUST land in the same feature scope as the code changes they describe.
- Before implementation is considered complete, the canonical automated command and
  any required targeted contract checks MUST be runnable from repo root.

## Governance

This constitution is the highest-priority project guidance for planning and
implementation artifacts in this repository. Specs, plans, tasks, docs, and reviews
MUST comply with it. Amendments require:

1. A written update to this file.
2. A semantic version decision with rationale.
3. Review of dependent planning templates and active feature artifacts.
4. An explicit note of any follow-up work that remains after the amendment.

Versioning policy:

- MAJOR: incompatible governance changes or principle removal/redefinition.
- MINOR: new principle or materially expanded governance requirement.
- PATCH: clarification without changing the governing meaning.

Compliance review expectation:

- Every planning pass MUST check constitution alignment.
- Every implementation review MUST treat constitution violations as blocking unless a
  documented amendment lands first.

**Version**: 1.0.0 | **Ratified**: 2026-03-21 | **Last Amended**: 2026-03-21