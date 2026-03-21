# Contract: Contributor Test Workflow

## Purpose

Define the supported contributor-facing workflow for installation, importability, and regression execution in this remediation.

## Supported Workflow

1. Create or sync the supported project environment for the current platform.
2. Install the project through the documented package workflow so `nanochat` is importable from that environment.
3. Run the canonical automated regression command: `python -m pytest -q`.

## Guarantees

1. The project package is importable from the supported environment without manual path overrides.
2. The canonical automated command collects tests without repository-root path assumptions.
3. The documented workflow is the same one used for local validation and automation.

## Non-Goals

1. Supporting multiple equivalent canonical test commands during this remediation.
2. Requiring contributors to export path overrides or depend on incidental transitive packages.

## Verification Expectations

1. A fresh environment following the documented setup can import `nanochat`.
2. `python -m pytest -q` executes as the repository’s standard automated test command.
3. Packaging metadata and documentation stay aligned with the supported workflow.
