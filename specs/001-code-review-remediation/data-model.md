# Data Model: Code Review Remediation Baseline

## Overview

This feature does not introduce persisted application data. Its design model is a set of repository contracts and verification objects that define acceptable contributor, runtime, and integration behavior.

## Entities

### Contributor Workflow

- Purpose: Represents the supported path for setting up the project, installing the package into the chosen environment, and running the canonical regression command.
- Fields:
  - environment_manager: named setup mechanism used by contributors
  - install_step: documented action that makes `nanochat` importable
  - canonical_test_command: single supported automated test invocation
  - expected_import_state: whether package import succeeds without path overrides
- Relationships:
  - Validates Regression Baseline
  - Depends on Dependency Declaration Set
- Validation rules:
  - Must define exactly one canonical test command for this feature
  - Must not require manual environment path overrides

### Regression Baseline

- Purpose: Represents the expected green state of the reviewed automated checks after remediation.
- Fields:
  - canonical_suite_status: pass/fail status for the supported default command
  - targeted_contract_checks: list of required focused validations
  - known_failure_count: count of remediation-scope failures remaining
- Relationships:
  - Verified by Contributor Workflow
  - Includes Message Role Contract, Runtime Precision Policy, Build Freshness Contract, and Telemetry Capability Contract checks
- Validation rules:
  - known_failure_count must be zero for remediation-scope failures
  - targeted contract checks must align with reviewed defect areas

### Message Role Contract

- Purpose: Defines which request message roles are accepted and how unsupported roles are rejected.
- Fields:
  - supported_roles: `{user, assistant}`
  - unsupported_role_examples: `system` and any other non-supported values
  - error_contract: validation response semantics for unsupported roles
- Relationships:
  - Included in Regression Baseline
  - Reflected in external contract documentation
- Validation rules:
  - Supported roles and error text must match exactly
  - Unsupported roles must not be silently ignored

### Runtime Precision Policy

- Purpose: Represents the authoritative runtime source for compute precision and dependent allocations.
- Fields:
  - default_precision_rule: hardware-aware default selection
  - override_mechanism: explicit user-set precision override
  - dependent_allocations: runtime structures that must follow the policy
- Relationships:
  - Included in Regression Baseline
  - Constrains inference cache allocation
- Validation rules:
  - Dependent allocations must not use a conflicting heuristic
  - Override behavior must propagate to covered runtime paths

### Telemetry Capability Contract

- Purpose: Defines the minimum backend capability needed to report runtime memory telemetry safely.
- Fields:
  - backend_presence: whether the accelerator backend exists
  - callable_capabilities: set of required telemetry callables
  - fallback_result: stable telemetry output when capability is absent or partial
- Relationships:
  - Included in Regression Baseline
- Validation rules:
  - Partial or mocked backends must return stable results rather than raising unexpected errors

### Build Freshness Contract

- Purpose: Defines when native helper outputs are considered current relative to their source inputs.
- Fields:
  - input_roots: authoritative source/manifests checked for freshness
  - output_roots: authoritative generated executable and companion outputs
  - rebuild_rule: condition under which a clean build is triggered
- Relationships:
  - Included in Regression Baseline
  - Referenced by native helper callers and tests
- Validation rules:
  - All callers must share the same output contract
  - Missing required outputs must force rebuild

### Dependency Declaration Set

- Purpose: Represents all direct dependencies required by supported runtime and evaluation paths.
- Fields:
  - declared_runtime_dependencies: packages listed in metadata
  - direct_runtime_imports: packages imported by covered runtime/evaluation paths
  - training_only_dependencies: packages required only for training-oriented flows
- Relationships:
  - Enables Contributor Workflow
  - Constrains Runtime Import Boundary
- Validation rules:
  - Every direct runtime import must be declared
  - Training-only dependencies must not be required by runtime-only flows

### Runtime Import Boundary

- Purpose: Distinguishes artifact-loading and inference flows from training-only module behavior.
- Fields:
  - runtime_paths: import and execution paths used by inference/evaluation workflows
  - training_paths: import and execution paths used only to produce training artifacts
  - lazy_load_points: boundaries where training-only tooling is loaded on demand
- Relationships:
  - Constrained by Dependency Declaration Set
- Validation rules:
  - Runtime paths must remain usable without training-only tooling unless the requested operation explicitly needs it
