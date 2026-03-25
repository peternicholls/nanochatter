# Remediation Addendum: Planning Consistency Fixes

## Purpose

Resolve the gaps found in the 2026-03-21 cross-artifact analysis before
implementation begins.

## Binding Decisions

1. Governance is no longer implicit. The repository constitution is now ratified at
   `.specify/memory/constitution.md`, and all feature artifacts for this remediation
   must satisfy it directly.
2. The supported contributor workflow is explicit:
   - sync dependencies with `uv sync --extra <platform>`
   - run the automated regression command with `uv run python -m pytest -q`
3. Automation must use the same canonical workflow as contributors. This remediation
   therefore includes an automation smoke-check task, not just local documentation.
4. User Story 2 and User Story 3 remain independently implementable through targeted
   contract checks, but final repo-wide validation uses the canonical workflow from
   User Story 1.
5. User Story 3 coverage explicitly includes evaluation-path dependency boundaries in
   addition to runtime-only imports.
6. Success criteria must be measurable in repo terms, not aspirational contributor
   percentages.

## Required Planning Changes

### Specification

- Name the canonical setup and regression commands explicitly.
- Add a requirement for contributor and automation workflow parity.
- Refine User Story 3 language to cover evaluation paths.
- Replace the ambiguous importability success criterion with a concrete smoke check.

### Plan

- Replace fallback governance language with constitution-based gates.
- Carry the canonical `uv` workflow through technical context and validation flow.
- Reference this addendum as a planning input.

### Tasks

- Add work to create an automation smoke check using the canonical workflow.
- Update runtime-boundary tasks to cover evaluation imports as well.
- Clarify dependency notes so story-local verification stays independent even though
  final quickstart validation uses User Story 1.

## Exit Condition

Planning is considered rerun successfully once `spec.md`, `plan.md`, `tasks.md`, and
the related validation docs all reflect the decisions above.