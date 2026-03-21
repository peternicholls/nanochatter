# Tasks: Code Review Remediation Baseline

**Input**: Design documents from `/specs/001-code-review-remediation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Regression and contract tests are required for this feature because the spec centers on restoring a trustworthy automated signal and validating documented workflow/runtime contracts.

**Organization**: Tasks are grouped by user story to preserve independently testable increments.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g. `[US1]`, `[US2]`, `[US3]`)
- Every task includes exact file paths in the description

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the shared scaffolding for remediation-focused regression work.

- [ ] T001 Add remediation packaging baseline in /Users/peternicholls/Dev/nanochatter/pyproject.toml
- [ ] T002 [P] Add shared remediation fixtures in /Users/peternicholls/Dev/nanochatter/tests/conftest.py
- [ ] T003 [P] Create remediation regression harness files in /Users/peternicholls/Dev/nanochatter/tests/test_packaging_workflow.py, /Users/peternicholls/Dev/nanochatter/tests/test_chat_web_validation.py, /Users/peternicholls/Dev/nanochatter/tests/test_runtime_boundaries.py, and /Users/peternicholls/Dev/nanochatter/tests/test_report.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define the shared regression contracts that all user stories rely on.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [ ] T004 Define canonical importability and test-command contract coverage in /Users/peternicholls/Dev/nanochatter/tests/test_packaging_workflow.py
- [ ] T005 [P] Define chat request role contract coverage in /Users/peternicholls/Dev/nanochatter/tests/test_chat_web_validation.py
- [ ] T006 [P] Define runtime-boundary and reporting contract coverage in /Users/peternicholls/Dev/nanochatter/tests/test_runtime_boundaries.py and /Users/peternicholls/Dev/nanochatter/tests/test_report.py

**Checkpoint**: Shared regression harness is ready; user story work can proceed.

---

## Phase 3: User Story 1 - Restore a Reliable Contributor Workflow (Priority: P1) 🎯 MVP

**Goal**: Make the package installable in the supported environment and standardize one canonical automated test workflow.

**Independent Test**: Install the project into a fresh supported environment, confirm `nanochat` imports without path overrides, and run `python -m pytest -q` successfully through collection.

### Tests for User Story 1

- [ ] T007 [US1] Add contributor workflow assertions in /Users/peternicholls/Dev/nanochatter/tests/test_packaging_workflow.py

### Implementation for User Story 1

- [ ] T008 [US1] Finalize build-system and package-install metadata in /Users/peternicholls/Dev/nanochatter/pyproject.toml
- [ ] T009 [US1] Document canonical install and test workflow in /Users/peternicholls/Dev/nanochatter/README.md
- [ ] T010 [US1] Align remediation validation steps in /Users/peternicholls/Dev/nanochatter/specs/001-code-review-remediation/quickstart.md

**Checkpoint**: User Story 1 should now be independently functional and verifiable.

---

## Phase 4: User Story 2 - Reestablish a Trustworthy Regression Signal (Priority: P2)

**Goal**: Fix the reviewed runtime and integration contract mismatches so the default regression signal is meaningful again.

**Independent Test**: Run the canonical regression command plus targeted tests for telemetry, native helper build freshness, chat role validation, and runtime precision behavior, and confirm they all pass with documented behavior.

### Tests for User Story 2

- [ ] T011 [P] [US2] Expand accelerator telemetry coverage in /Users/peternicholls/Dev/nanochatter/tests/test_common_mps_memory.py
- [ ] T012 [P] [US2] Expand native helper freshness coverage in /Users/peternicholls/Dev/nanochatter/tests/test_swift_stub_engine.py
- [ ] T013 [P] [US2] Add chat role validation coverage in /Users/peternicholls/Dev/nanochatter/tests/test_chat_web_validation.py
- [ ] T014 [P] [US2] Add runtime precision override coverage in /Users/peternicholls/Dev/nanochatter/tests/test_engine.py

### Implementation for User Story 2

- [ ] T015 [US2] Harden accelerator telemetry helpers in /Users/peternicholls/Dev/nanochatter/nanochat/common.py
- [ ] T016 [US2] Unify native helper build freshness paths in /Users/peternicholls/Dev/nanochatter/nanochat/swift_build.py and /Users/peternicholls/Dev/nanochatter/scripts/mlx_swift_stub.py
- [ ] T017 [US2] Enforce the user/assistant-only role contract in /Users/peternicholls/Dev/nanochatter/scripts/chat_web.py
- [ ] T018 [US2] Derive KV-cache dtype from runtime precision policy in /Users/peternicholls/Dev/nanochatter/nanochat/engine.py

**Checkpoint**: User Story 2 should now be independently functional and verifiable.

---

## Phase 5: User Story 3 - Reduce Runtime Surprise from Dependencies and Side Effects (Priority: P3)

**Goal**: Make runtime imports, dependency metadata, initialization behavior, and reporting diagnostics align with the documented runtime boundary.

**Independent Test**: Build the supported environment from metadata, exercise runtime-only import/load paths without training-only tooling, and verify report failures are diagnosable.

### Tests for User Story 3

- [ ] T019 [P] [US3] Add runtime import-boundary coverage in /Users/peternicholls/Dev/nanochatter/tests/test_runtime_boundaries.py
- [ ] T020 [P] [US3] Add report diagnostics coverage in /Users/peternicholls/Dev/nanochatter/tests/test_report.py

### Implementation for User Story 3

- [ ] T021 [US3] Declare direct runtime dependencies in /Users/peternicholls/Dev/nanochatter/pyproject.toml
- [ ] T022 [US3] Lazy-load training-only tokenization imports in /Users/peternicholls/Dev/nanochatter/nanochat/tokenizer.py and /Users/peternicholls/Dev/nanochatter/nanochat/checkpoint_manager.py
- [ ] T023 [US3] Reduce import-time logging side effects in /Users/peternicholls/Dev/nanochatter/nanochat/common.py and /Users/peternicholls/Dev/nanochatter/nanochat/checkpoint_manager.py
- [ ] T024 [US3] Surface diagnosable command failures in /Users/peternicholls/Dev/nanochatter/nanochat/report.py

**Checkpoint**: User Story 3 should now be independently functional and verifiable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Reconcile documentation and run end-to-end validation across all stories.

- [ ] T025 [P] Reconcile contributor workflow docs in /Users/peternicholls/Dev/nanochatter/README.md and /Users/peternicholls/Dev/nanochatter/specs/001-code-review-remediation/contracts/contributor-test-workflow.md
- [ ] T026 [P] Reconcile role-contract docs in /Users/peternicholls/Dev/nanochatter/specs/001-code-review-remediation/spec.md and /Users/peternicholls/Dev/nanochatter/specs/001-code-review-remediation/contracts/chat-role-validation.md
- [ ] T027 Run quickstart validation from /Users/peternicholls/Dev/nanochatter/specs/001-code-review-remediation/quickstart.md against /Users/peternicholls/Dev/nanochatter/tests/test_packaging_workflow.py, /Users/peternicholls/Dev/nanochatter/tests/test_common_mps_memory.py, /Users/peternicholls/Dev/nanochatter/tests/test_swift_stub_engine.py, /Users/peternicholls/Dev/nanochatter/tests/test_chat_web_validation.py, /Users/peternicholls/Dev/nanochatter/tests/test_engine.py, /Users/peternicholls/Dev/nanochatter/tests/test_runtime_boundaries.py, and /Users/peternicholls/Dev/nanochatter/tests/test_report.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Starts after Foundational completion.
- **User Story 2 (Phase 4)**: Starts after Foundational completion; verification uses the canonical workflow established by User Story 1.
- **User Story 3 (Phase 5)**: Starts after Foundational completion; verification uses the canonical workflow established by User Story 1.
- **Polish (Phase 6)**: Depends on completion of the user stories being shipped.

### User Story Dependencies

- **User Story 1 (P1)**: No dependency on other user stories once Foundational tasks are complete.
- **User Story 2 (P2)**: No code dependency on User Story 1, but final verification assumes the canonical contributor workflow from User Story 1.
- **User Story 3 (P3)**: No code dependency on User Story 2, but final verification assumes the canonical contributor workflow from User Story 1.

### Within Each User Story

- Regression tests and contract coverage should be written or updated before implementation tasks in the same story.
- Package and metadata changes must land before documentation claims that depend on them.
- Runtime contract fixes must land before final quickstart validation.

### Parallel Opportunities

- `T002` and `T003` can run in parallel during Setup.
- `T005` and `T006` can run in parallel during Foundational work.
- `T011`, `T012`, `T013`, and `T014` can run in parallel for User Story 2.
- `T019` and `T020` can run in parallel for User Story 3.
- `T025` and `T026` can run in parallel during Polish.

---

## Parallel Example: User Story 2

```bash
# Launch the User Story 2 regression coverage updates together:
Task: "Expand accelerator telemetry coverage in /Users/peternicholls/Dev/nanochatter/tests/test_common_mps_memory.py"
Task: "Expand native helper freshness coverage in /Users/peternicholls/Dev/nanochatter/tests/test_swift_stub_engine.py"
Task: "Add chat role validation coverage in /Users/peternicholls/Dev/nanochatter/tests/test_chat_web_validation.py"
Task: "Add runtime precision override coverage in /Users/peternicholls/Dev/nanochatter/tests/test_engine.py"
```

---

## Parallel Example: User Story 3

```bash
# Launch the User Story 3 test coverage updates together:
Task: "Add runtime import-boundary coverage in /Users/peternicholls/Dev/nanochatter/tests/test_runtime_boundaries.py"
Task: "Add report diagnostics coverage in /Users/peternicholls/Dev/nanochatter/tests/test_report.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. Stop and validate package importability plus the canonical `python -m pytest -q` workflow.

### Incremental Delivery

1. Finish Setup and Foundational phases to establish the shared regression harness.
2. Deliver User Story 1 to stabilize contributor setup and the canonical test workflow.
3. Deliver User Story 2 to restore a green regression signal for reviewed contracts.
4. Deliver User Story 3 to reduce runtime surprise from dependencies and import behavior.
5. Finish with the Polish phase and quickstart validation.

### Parallel Team Strategy

1. One developer can finish Setup while another prepares shared regression fixtures.
2. After Foundational tasks are complete, User Story 2 and User Story 3 test updates can proceed in parallel.
3. Documentation reconciliation can proceed in parallel once implementation is stable.

---

## Notes

- All tasks follow the required checklist format and include exact file paths.
- `[P]` markers are used only where tasks can be executed in parallel without touching the same incomplete file set.
- User story tasks preserve independent verification, even though final repo validation depends on the canonical contributor workflow from User Story 1.
- `tasks.md` is intentionally scoped to the reviewed remediation items and avoids unrelated refactors.