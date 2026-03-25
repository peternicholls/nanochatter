from pathlib import Path

from nanochat import report as report_mod


def test_generate_header_surfaces_command_diagnostics(monkeypatch):
    def fake_run_command(cmd):
        if cmd.startswith("git rev-parse"):
            return {
                "cmd": cmd,
                "ok": False,
                "stdout": "",
                "stderr": "fatal: not a git repository",
                "returncode": 128,
                "error": None,
            }
        return {
            "cmd": cmd,
            "ok": True,
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "error": None,
        }

    monkeypatch.setattr(report_mod, "run_command", fake_run_command)

    header = report_mod.generate_header()

    assert "### Diagnostics" in header
    assert "git rev-parse --short HEAD" in header
    assert "fatal: not a git repository" in header


def test_report_generate_surfaces_timestamp_parse_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "header.md").write_text(
        "# nanochat training report\n\nRun started: not-a-timestamp\n\n---\n\n",
        encoding="utf-8",
    )

    report = report_mod.Report(str(report_dir))
    report_file = Path(report.generate())
    content = report_file.read_text(encoding="utf-8")

    assert "## Diagnostics" in content
    assert "Failed to parse start timestamp from header.md" in content