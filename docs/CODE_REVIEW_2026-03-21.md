# Code Review

Date: 2026-03-21

## Scope

This review covered the current repository head with emphasis on the core `nanochat/` package, the main entry scripts, the Swift/MLX integration, packaging/test ergonomics, and the existing automated tests.

Validation performed during the review:

```bash
pytest -q
PYTHONPATH=. pytest -q
./.venv/bin/pytest -q
./.venv/bin/python -m pytest -q
./.venv/bin/python - <<'PY'
import sys, importlib.util
sys.path = [p for p in sys.path if p]
print(importlib.util.find_spec("nanochat"))
PY
```

Observed results:

- `pytest -q` failed during collection with `ModuleNotFoundError: nanochat`
- `PYTHONPATH=. pytest -q` progressed further but then failed collection because `rustbpe` was unavailable in that interpreter
- `./.venv/bin/pytest -q` still failed collection with `ModuleNotFoundError: nanochat`
- `./.venv/bin/python -m pytest -q` was the only reliable invocation in this environment and produced `33 passed, 10 skipped, 3 failed`
- removing the repo root from `sys.path` made `find_spec("nanochat")` return `None`, confirming the project is not installed into the virtualenv as a package

## Needs Fixes

1. The default test workflow is brittle and not reliably runnable from the project virtualenv.
   - Evidence: `./.venv/bin/pytest -q` fails to import `nanochat`, while `./.venv/bin/python -m pytest -q` works.
   - References: `pyproject.toml:1`, `pyproject.toml:34`
   - Impact: contributors can follow a normal Python workflow and get collection failures unrelated to the code under test.
   - Recommendation: add an explicit build backend/package install path, or otherwise make the package importable without depending on the repo root being present on `sys.path`. At minimum, standardize one canonical test command and enforce it in CI.

2. The current test suite is red on `main`.
   - Evidence: `./.venv/bin/python -m pytest -q` ends with 3 failures.
   - References: `nanochat/common.py:38`, `nanochat/common.py:83`, `nanochat/swift_build.py:75`, `tests/test_common_mps_memory.py:22`, `tests/test_swift_stub_engine.py:233`
   - Impact: the repository currently lacks a clean regression signal.
   - Recommendation:
     - fix `get_mps_memory_stats()` / `is_mps_available()` so the MPS telemetry helper is robust to the test shim used in `tests/test_common_mps_memory.py`
     - reconcile the `xcodebuild -derivedDataPath` contract between `nanochat/swift_build.py` and `tests/test_swift_stub_engine.py`

3. `scripts/chat_web.py` exposes an inconsistent public API contract for message roles.
   - Evidence: validation rejects any role other than `user` or `assistant`, but the error message says `system` is valid, and the tokenization path also ignores `system`.
   - References: `scripts/chat_web.py:184`, `scripts/chat_web.py:186`, `scripts/chat_web.py:331`
   - Impact: clients cannot tell whether system prompts are supported, rejected intentionally, or silently dropped.
   - Recommendation: either implement `system` end-to-end or remove it from the contract and error text. Add request-validation tests for the chosen behavior.

4. The inference engine hardcodes KV-cache dtype from device type instead of the repo’s configured compute dtype.
   - Evidence: `Engine.generate()` picks `bfloat16` for CUDA and `float32` otherwise, even though `nanochat.common` already exposes `COMPUTE_DTYPE`.
   - References: `nanochat/common.py:22`, `nanochat/common.py:36`, `nanochat/engine.py:174`, `nanochat/engine.py:180`
   - Impact: forced CUDA modes such as `NANOCHAT_DTYPE=float16` or `float32` can drift from cache allocation dtype and create correctness or performance surprises.
   - Recommendation: make KV-cache dtype derive from one authoritative runtime source, then add inference tests that cover non-default dtype overrides.

## Needs Improvements

1. Optional dependency boundaries are still too broad in the tokenizer/loading path.
   - Evidence: `rustbpe` is imported at module load time even though runtime tokenizer loading uses serialized artifacts, and `engine.py` imports `load_model`, which imports `get_tokenizer`, which imports `rustbpe`.
   - References: `nanochat/engine.py:21`, `nanochat/checkpoint_manager.py:13`, `nanochat/tokenizer.py:158`
   - Impact: unrelated runtime and test codepaths inherit training-time dependencies.
   - Recommendation: lazy-import training-only tokenizer dependencies and reduce import-time coupling between inference helpers and checkpoint/tokenizer construction.

2. Several direct runtime dependencies are used but not declared explicitly.
   - Evidence: the code imports `filelock`, `requests`, `pyarrow`, `yaml`, and `jinja2`, but they are not listed in `pyproject.toml`.
   - References: `pyproject.toml:7`, `nanochat/common.py:11`, `nanochat/dataset.py:13`, `nanochat/dataset.py:14`, `scripts/base_eval.py:26`, `nanochat/core_eval.py:10`
   - Impact: fresh environments work only because transitive dependencies happen to provide the missing packages.
   - Recommendation: declare all direct imports explicitly and keep environment creation deterministic.

3. Import-time side effects remain heavier than they should be.
   - Evidence: `nanochat.common` freezes compute dtype and configures logging during import; `checkpoint_manager` calls logging setup again.
   - References: `nanochat/common.py:22`, `nanochat/common.py:67`, `nanochat/common.py:75`, `nanochat/checkpoint_manager.py:14`, `nanochat/checkpoint_manager.py:17`
   - Impact: behavior depends on import order, and library reuse/testing is harder than necessary.
   - Recommendation: move logging and runtime initialization behind explicit setup functions.

4. `scripts/chat_web.py` has a CORS configuration that is both overly broad and internally inconsistent.
   - Evidence: it uses `allow_origins=["*"]` together with `allow_credentials=True`.
   - References: `scripts/chat_web.py:227`
   - Impact: this is not a sound long-term configuration for a public API and can behave unexpectedly in browsers once credentials are involved.
   - Recommendation: narrow allowed origins or disable credentials if they are not needed.

5. Reporting utilities still hide too much operational detail.
   - Evidence: `nanochat/report.py` uses `shell=True` and broad exception swallowing in command helpers and parsing paths.
   - References: `nanochat/report.py:15`, `nanochat/report.py:18`, `nanochat/report.py:25`, `nanochat/report.py:255`
   - Impact: report generation failures can silently degrade output and become hard to diagnose.
   - Recommendation: switch to argument lists where practical, catch specific exceptions, and emit warnings instead of silently returning `None`.

## What Is Good

1. The repository is still structurally readable.
   - The separation between `nanochat/` library code, `scripts/` entrypoints, `tasks/`, `runs/`, and `tests/` is easy to navigate.

2. The test suite has meaningful coverage around high-risk inference paths.
   - Coverage includes engine behavior, attention fallback parity, Swift stub routing/build behavior, MLX compile utilities, and MPS memory helpers.
   - References: `tests/test_engine.py:1`, `tests/test_attention_fallback.py:1`, `tests/test_swift_stub_engine.py:1`, `tests/test_common_mps_memory.py:1`

3. The dataset download path shows good operational hygiene.
   - Downloads use retries, exponential backoff, and temp-file writes before final rename.
   - References: `nanochat/dataset.py:110`, `nanochat/dataset.py:117`, `nanochat/dataset.py:127`

4. The web API includes practical abuse-prevention limits.
   - Request size, message size, temperature, top-k, and max-token limits are validated server-side before generation.
   - References: `scripts/chat_web.py:51`, `scripts/chat_web.py:153`

5. Swift build logic has at least been partially centralized instead of being duplicated ad hoc everywhere.
   - `nanochat/swift_build.py` is a good direction for keeping build freshness rules and output paths in one place.
   - References: `nanochat/swift_build.py:1`

## Overall Assessment

The codebase remains workable and comparatively readable for this problem space, and the inference-related tests add real value. The immediate problem is trust in the development workflow: the default test invocation is brittle, the suite is currently red, and there are still a few contract mismatches at API and integration boundaries. After that, the highest-value cleanup is reducing import-time coupling and making dependency/runtime configuration more explicit.
