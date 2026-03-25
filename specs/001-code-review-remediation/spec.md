# Feature Specification: Code Review Remediation Baseline

**Feature Branch**: `[001-code-review-remediation]`  
**Created**: 2026-03-21  
**Status**: Draft  
**Input**: User description: "create a spec based on this code review. keep it within this branch"

## Clarifications

### Session 2026-03-21

- Q: Should the chat request interface support `system` role messages during this remediation, or reject them? → A: Reject `system` for now; only `user` and `assistant` are supported roles in this remediation.
- Q: What exact contributor workflow should this remediation standardize? → A: Use `uv sync --extra <platform>` as the setup/install workflow and `uv run python -m pytest -q` as the canonical automated regression command.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Restore a Reliable Contributor Workflow (Priority: P1)

As a contributor, I need a predictable project setup and test entrypoint so that I can install the repository, import the package, and run the expected regression checks without environment-specific workarounds.

**Why this priority**: The audit identified the development workflow itself as unreliable. Until package import and the standard test path work consistently, every other fix remains harder to validate and contributors cannot trust failures.

**Independent Test**: From repo root on a supported platform, run `uv sync --extra <platform>`, confirm `uv run python -c "import nanochat"` succeeds, and run `uv run python -m pytest -q` without relying on manual environment path overrides.

**Acceptance Scenarios**:

1. **Given** a contributor is at repo root on a supported platform, **When** they run `uv sync --extra <platform>`, **Then** `uv run python -c "import nanochat"` succeeds without manual path overrides.
2. **Given** a contributor follows the documented setup workflow, **When** they run `uv run python -m pytest -q`, **Then** test collection starts without package import failures caused by repository path assumptions.
3. **Given** the repository documents one standard test workflow, **When** contributors and automation follow that workflow, **Then** they receive the same expected baseline result.

---

### User Story 2 - Reestablish a Trustworthy Regression Signal (Priority: P2)

As a maintainer, I need the default regression suite to pass and the reviewed interface and integration contracts to behave consistently so that a green test run means the repository is safe to change.

**Why this priority**: The audit found active failures and contract mismatches in accelerator telemetry, native build behavior, chat message roles, and runtime precision handling. These are direct correctness issues that undermine confidence in the current branch.

**Independent Test**: Run `uv run python -m pytest -q` and the targeted contract checks for accelerator telemetry, native build freshness, chat request validation, and runtime precision behavior; the suite should finish green and reflect the documented contracts.

**Acceptance Scenarios**:

1. **Given** the repository's supported default test environment, **When** the standard automated test suite is executed, **Then** it completes without the currently known remediation-scope failures.
2. **Given** runtime telemetry is requested on systems without an active accelerator backend or with test doubles, **When** memory statistics are gathered, **Then** the helper returns a stable, well-defined result instead of failing on backend shape assumptions.
3. **Given** the native helper build outputs are fresh or stale relative to their inputs, **When** build freshness is evaluated, **Then** the build behavior matches the documented and tested output location contract.
4. **Given** a chat completion request includes `user`, `assistant`, or any unsupported role such as `system`, **When** the request is validated, **Then** only `user` and `assistant` are accepted and unsupported roles are rejected with a matching error contract.
5. **Given** inference runs under a configured compute precision override, **When** generation allocates cache state, **Then** the cache precision follows the authoritative runtime precision policy rather than a separate device-only rule.

---

### User Story 3 - Reduce Runtime Surprise from Dependencies and Side Effects (Priority: P3)

As a maintainer, I need runtime dependencies and import behavior to match actual product usage so that inference, evaluation, and tooling paths do not break because of hidden training-only requirements or import-order side effects.

**Why this priority**: The audit shows that direct dependencies are incompletely declared and that runtime code currently inherits unnecessary import-time coupling. This raises setup costs and creates failures that are unrelated to the task a contributor is trying to perform.

**Independent Test**: Inspect the declared dependency set, exercise representative runtime and evaluation paths, and confirm those paths avoid unrelated training-only requirements while documented runtime initialization happens predictably.

**Acceptance Scenarios**:

1. **Given** a fresh environment built from the declared project dependencies, **When** a supported runtime or evaluation path is executed, **Then** it does not depend on undeclared direct packages being supplied transitively.
2. **Given** an inference-oriented or checkpoint-loading path that does not train tokenization assets, **When** it is imported and used, **Then** it does not fail solely because training-only tokenization tooling is missing.
3. **Given** the repository is imported by tests or library consumers, **When** initialization occurs, **Then** logging and runtime configuration behavior is explicit and stable rather than heavily dependent on import order.
4. **Given** operational tooling encounters command or parsing failures, **When** report generation continues, **Then** the failure is surfaced clearly enough for maintainers to diagnose degraded output.

### Edge Cases

- A contributor uses the documented `uv` workflow on a supported platform extra, but the package is still not importable from the resulting environment.
- Memory telemetry is queried on systems where the accelerator backend is absent, partially stubbed, or exposes only a subset of expected attributes.
- The native helper executable exists but a required companion output is missing, or source inputs are newer than only one output artifact.
- A chat request includes `system` or another semantically plausible role that is not supported by the current conversation format.
- A compute precision override is set to a non-default value that differs from the device family’s usual default.
- A runtime-only code path is executed in an environment that intentionally omits training-only tooling.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The project MUST document `uv sync --extra <platform>` as the supported setup/install workflow that results in the `nanochat` package being importable from the repo-local project environment.
- **FR-002**: The project MUST provide one documented canonical automated test command for contributors and automation: `uv run python -m pytest -q`.
- **FR-003**: The canonical automated test workflow MUST collect tests without requiring contributors to apply manual environment path overrides.
- **FR-004**: The remediation baseline MUST eliminate the currently known test failures identified in the 2026-03-21 code review scope.
- **FR-005**: Runtime memory telemetry MUST handle unavailable or partially stubbed accelerator backends without raising unexpected errors.
- **FR-006**: Native helper build freshness evaluation MUST use a single consistent contract for input locations, output locations, and rebuild decisions.
- **FR-007**: The chat request interface MUST support only `user` and `assistant` roles for this remediation, and its validation errors MUST match that supported set exactly.
- **FR-008**: Inference cache allocation MUST derive precision from the repository’s authoritative runtime compute precision policy.
- **FR-009**: Directly imported runtime and evaluation dependencies MUST be declared explicitly in the project dependency metadata.
- **FR-010**: Runtime, inference-oriented, and evaluation code paths MUST avoid failing solely because training-only tokenization tooling is unavailable, unless the requested operation explicitly requires that tooling.
- **FR-011**: Runtime initialization side effects that affect logging or execution policy MUST be explicit, predictable, and testable.
- **FR-012**: Operational reporting helpers MUST expose command and parsing failures clearly enough for maintainers to diagnose degraded reports.
- **FR-013**: The repository MUST provide an automation smoke check that runs the same canonical setup and regression workflow used by contributors.

### Key Entities *(include if feature involves data)*

- **Contributor Workflow**: The documented setup, installation, import, and test-running path a maintainer or contributor follows in a supported environment.
- **Regression Baseline**: The expected result of the canonical automated test workflow, including collection success and resolution of currently known remediation-scope failures.
- **Message Role Contract**: The allowed conversation role values for this remediation (`user` and `assistant`), their validation behavior, and the corresponding error contract returned for unsupported roles such as `system`.
- **Runtime Precision Policy**: The single source of truth that defines which compute precision a runtime session uses under default detection and explicit overrides.
- **Dependency Boundary**: The distinction between dependencies required for runtime or evaluation flows and dependencies only needed for training or artifact creation.
- **Build Freshness Contract**: The rules that determine when native helper outputs are considered current relative to source inputs.

## Assumptions

- This feature is limited to the issues raised in the 2026-03-21 code review and does not attempt unrelated architectural refactors.
- The remediation should prefer clarifying and enforcing current supported behavior over expanding product scope unless expansion is required to remove an inconsistency.
- One canonical contributor workflow is preferable to multiple partially supported test entrypoints.
- The canonical contributor workflow is `uv`-based and should remain runnable from repo root without requiring manual environment activation steps in docs.
- Unsupported chat roles are rejected rather than added during this remediation so that the fix remains a consistency repair instead of a conversation-model expansion.
- Success is measured by restored contributor trust, a clean regression signal, and reduced setup/runtime surprise rather than by adding new end-user capabilities.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In fresh-environment validation, `uv sync --extra <platform>` followed by `uv run python -c "import nanochat"` succeeds for every supported platform profile explicitly covered by this feature.
- **SC-002**: The documented canonical automated test command `uv run python -m pytest -q` completes with zero known remediation-scope failures in the supported default environment.
- **SC-003**: 100% of reviewed contract checks for accelerator telemetry, native helper freshness, chat role validation, and runtime precision selection behave consistently with the documented feature behavior, including rejection of unsupported roles such as `system`.
- **SC-004**: A fresh environment created from declared project dependencies can execute the supported runtime and evaluation paths covered by this feature without failure caused by missing direct dependencies.
- **SC-005**: Runtime-only paths no longer require contributors to install training-only tokenization tooling unless they explicitly invoke a training-oriented workflow.
- **SC-006**: Maintainers can identify degraded report-generation behavior from surfaced error output in every command or parsing failure scenario covered by this feature.
- **SC-007**: One automation smoke check runs the same canonical `uv` setup and regression workflow that the contributor docs prescribe.
