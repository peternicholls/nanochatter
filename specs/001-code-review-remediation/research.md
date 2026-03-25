# Research: Code Review Remediation Baseline

## Decision 1: Add an explicit Python build backend and standardize one canonical `uv` workflow

- Decision: Add an explicit `[build-system]` section for the existing Python package, standardize `uv sync --extra <platform>` as the documented setup/install workflow, and standardize the automated regression command as `uv run python -m pytest -q`.
- Rationale: The review evidence shows the project package is not installed into the virtual environment today, which breaks direct pytest entrypoints. An explicit build backend is the minimal standards-compliant way to make the existing package installable. Using the `uv` sync-and-run path gives contributors and automation the same deterministic repo-root workflow without depending on prior environment activation.
- Alternatives considered:
  - Rely on repository-root path injection: rejected because it preserves a brittle, undocumented import assumption.
  - Standardize on the bare `pytest` executable: rejected because that path is the one currently failing and is less robust during the transition.
  - Introduce a new wrapper script for tests: rejected because a metadata fix plus one canonical command is simpler.

## Decision 2: Reject `system` role messages during this remediation

- Decision: Keep the chat request contract limited to `user` and `assistant` and reject `system` explicitly.
- Rationale: The feature scope is remediation, not prompt-model expansion. Rejecting `system` removes the contradiction between validation and runtime behavior with the smallest behavioral change and without introducing new conversation semantics.
- Alternatives considered:
  - Add full `system` support end-to-end: rejected because it expands product behavior beyond the bug-fix scope.
  - Accept but ignore `system`: rejected because silently discarding meaningful user input is a weaker contract than explicit rejection.

## Decision 3: Treat native helper build outputs as a single shared contract rooted under `swift/Build`

- Decision: Keep one authoritative native helper build contract centered on `xcodebuild -derivedDataPath <repo>/swift/Build`, and make tests and helper callers agree with that shared path logic.
- Rationale: The repo already centralizes most of the build logic in `nanochat/swift_build.py`. Aligning every caller and test to that single contract is simpler and less error-prone than allowing multiple build-output conventions.
- Alternatives considered:
  - Move build products to a new derived-data location used only by tests: rejected because it would split the contract.
  - Let each caller compute its own output paths: rejected because the repo is already partway through centralization and divergence caused the current mismatch.

## Decision 4: Make runtime telemetry tolerant of absent or partially stubbed accelerator backends

- Decision: Gate runtime telemetry on backend availability and required callable attributes before invoking backend-specific memory functions.
- Rationale: The reviewed failure comes from assuming a full accelerator backend shape. Defensive capability checks preserve expected output on real systems and allow test doubles to model only the behavior each test needs.
- Alternatives considered:
  - Require tests to fully emulate backend internals: rejected because the production helper should already be robust to absent or partial capability.
  - Disable telemetry tests when the backend is mocked: rejected because that would remove valuable coverage from a reviewed high-risk path.

## Decision 5: Use the repo-wide runtime precision policy as the only cache precision source

- Decision: Make inference cache allocation derive from the authoritative runtime compute precision policy rather than a separate device-type heuristic.
- Rationale: The repo already has a global precision policy and explicit override mechanism. Cache allocation is part of runtime compute behavior and should not drift from the same source of truth.
- Alternatives considered:
  - Keep the current CUDA/non-CUDA heuristic and document it: rejected because it conflicts with existing override behavior.
  - Introduce a second cache-specific precision setting: rejected because it creates more policy surface for little gain.

## Decision 6: Separate training-only tokenization tooling from runtime import paths

- Decision: Lazy-load training-only tokenization dependencies and keep runtime tokenization-from-artifact paths usable without those training tools.
- Rationale: Runtime and checkpoint-loading flows should not fail because a training dependency was imported at module load time. Lazy import at the exact training entrypoint is the smallest separation that fixes this boundary.
- Alternatives considered:
  - Keep training-only tools as always-on runtime dependencies: rejected because it preserves unnecessary setup burden and import failures.
  - Split tokenizer code into a new package immediately: rejected because that is a broader refactor than this remediation requires.

## Decision 7: Declare all directly imported runtime and evaluation dependencies explicitly in project metadata

- Decision: Add explicit dependency metadata for every package imported directly by supported runtime and evaluation paths.
- Rationale: Deterministic environment creation is part of the contributor workflow requirement. Direct imports should not rely on accidental transitive availability.
- Alternatives considered:
  - Keep depending on transitive installation: rejected because it is fragile and environment-specific.
  - Move all optional imports behind runtime checks without declaring them: rejected because supported paths still need those packages declared.

## Decision 8: Keep import-time side-effect cleanup scoped and incremental

- Decision: Limit this feature to making initialization behavior predictable and idempotent in touched paths rather than redesigning every global runtime policy.
- Rationale: The spec requires explicit and testable initialization behavior, but a full inversion of all import-time state would expand the remediation beyond the reviewed defects. The smallest compliant approach is to reduce duplicate or surprising side effects where the audited seams already touch them.
- Alternatives considered:
  - Full runtime initialization redesign: rejected as out of scope for this remediation.
  - Ignore side effects entirely: rejected because the review identifies them as a contributor and testability problem.

## Decision 9: Replace silent operational-report failures with diagnosable behavior

- Decision: Prefer explicit subprocess argument lists and targeted failure reporting in reporting helpers while preserving report generation where partial output is still possible.
- Rationale: Report generation is operational tooling. The right remediation is not to fail noisily on every partial error, but to surface enough information for maintainers to understand degraded output.
- Alternatives considered:
  - Keep broad `shell=True` and silent fallback behavior: rejected because it hides the root cause of degraded reports.
  - Make report generation hard-fail on every command issue: rejected because partial reports still have value.
