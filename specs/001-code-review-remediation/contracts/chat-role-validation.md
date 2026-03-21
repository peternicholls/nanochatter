# Contract: Chat Role Validation

## Purpose

Define the externally visible request-role behavior for the remediation feature.

## Supported Roles

- `user`
- `assistant`

## Unsupported Roles

- `system`
- Any other role not explicitly listed as supported

## Request Validation Rules

1. A chat request is valid only when every message role is one of the supported roles.
2. If any message role is unsupported, the request is rejected.
3. Rejection behavior must match the documented supported-role set exactly.
4. Unsupported roles are never silently ignored or reinterpreted in this remediation.

## Verification Expectations

1. A request containing only `user` and `assistant` roles passes validation.
2. A request containing `system` is rejected.
3. The rejection message names only the roles that are actually supported.
4. Runtime tokenization and validation behavior remain aligned.
