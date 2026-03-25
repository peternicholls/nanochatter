from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
QUICKSTART = ROOT / "specs" / "001-code-review-remediation" / "quickstart.md"
WORKFLOW = ROOT / ".github" / "workflows" / "remediation-smoke.yml"


def test_pyproject_declares_build_backend_and_direct_runtime_dependencies():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert data["build-system"] == {
        "requires": ["setuptools>=80.9.0"],
        "build-backend": "setuptools.build_meta",
    }
    assert data["tool"]["setuptools"]["packages"] == ["nanochat"]

    dependencies = set(data["project"]["dependencies"])
    expected = {
        "filelock>=3.19.1",
        "jinja2>=3.1.6",
        "pyarrow>=21.0.0",
        "pyyaml>=6.0.3",
        "requests>=2.32.5",
    }
    assert expected.issubset(dependencies)


def test_docs_and_workflow_share_canonical_uv_commands():
    expected_doc_lines = [
        "uv sync --extra gpu",
        "uv sync --extra cpu",
        "uv sync --extra macos",
        'uv run python -c "import nanochat"',
        "uv run python -m pytest -q",
    ]

    for path in (README, QUICKSTART):
        content = path.read_text(encoding="utf-8")
        for expected in expected_doc_lines:
            assert expected in content, f"{expected} missing from {path}"

    workflow_content = WORKFLOW.read_text(encoding="utf-8")
    assert "uv sync --extra cpu --group dev" in workflow_content
    assert 'uv run python -c "import nanochat"' in workflow_content
    assert "uv run python -m pytest -q" in workflow_content


def test_package_imports_from_repo_root():
    import nanochat

    assert nanochat is not None