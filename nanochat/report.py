"""
Utilities for generating training report cards. More messy code than usual, will fix.
"""

import os
import re
import shutil
import subprocess
import socket
import datetime
import platform
import psutil
import torch

def run_command(cmd):
    """Run a shell command and return structured diagnostics."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return {
            "cmd": cmd,
            "ok": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "error": None,
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "ok": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error": str(exc),
        }

def command_output(result):
    if result["stdout"]:
        return result["stdout"]
    if result["ok"]:
        return ""
    return None

def command_failure_message(result):
    details = []
    if result["returncode"] is not None:
        details.append(f"exit={result['returncode']}")
    if result["stderr"]:
        details.append(f"stderr={result['stderr']}")
    if result["error"]:
        details.append(f"error={result['error']}")
    suffix = "; ".join(details) if details else "no additional details"
    return f"Command failed: {result['cmd']} ({suffix})"

def get_git_info():
    """Get current git commit, branch, and dirty status."""
    info = {"diagnostics": []}
    commit_result = run_command("git rev-parse --short HEAD")
    branch_result = run_command("git rev-parse --abbrev-ref HEAD")
    status_result = run_command("git status --porcelain")
    message_result = run_command("git log -1 --pretty=%B")

    info['commit'] = command_output(commit_result) or "unknown"
    info['branch'] = command_output(branch_result) or "unknown"

    # Check if repo is dirty (has uncommitted changes)
    status = command_output(status_result)
    info['dirty'] = bool(status) if status is not None else False

    # Get commit message
    info['message'] = command_output(message_result) or ""
    info['message'] = info['message'].split('\n')[0][:80]  # First line, truncated

    for result in (commit_result, branch_result, status_result, message_result):
        if not result["ok"]:
            info["diagnostics"].append(command_failure_message(result))

    return info

def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chip_name = platform.processor() or "Apple Silicon"
            return {
                "available": True,
                "backend": "mps",
                "count": 1,
                "names": [chip_name],
                "memory_gb": [psutil.virtual_memory().total / (1024**3)],
            }
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "backend": "cuda",
        "count": num_devices,
        "names": [],
        "memory_gb": []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))

    # Get CUDA version
    info["cuda_version"] = torch.version.cuda or "unknown"

    return info

def get_system_info():
    """Get system information."""
    info = {}

    # Basic system info
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU and memory
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # User and environment
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

def estimate_cost(gpu_info, runtime_hours=None):
    """Estimate training cost based on GPU type and runtime."""

    # Rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # Try to identify GPU type from name
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info["count"]
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info["count"]  # Default estimate

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None
    }

def generate_header():
    """Generate the header for a training report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)
    diagnostics = list(git_info.pop("diagnostics", []))

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Message: {git_info['message']}

### Hardware
- Platform: {sys_info['platform']}
- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- Memory: {sys_info['memory_gb']:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        if gpu_info.get("backend") == "cuda":
            header += f"""- GPUs: {gpu_info['count']}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info['cuda_version']}
"""
        else:
            header += f"""- Accelerator: {gpu_names} via MPS
- Unified Memory: {total_vram:.1f} GB total
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate: ${cost_info['hourly_rate']:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

"""

    # bloat metrics: count lines/chars in git-tracked source files only
    extensions = ['py', 'md', 'rs', 'html', 'toml', 'sh']
    git_patterns = ' '.join(f"'*.{ext}'" for ext in extensions)
    files_result = run_command(f"git ls-files -- {git_patterns}")
    files_output = command_output(files_result)
    file_list = [f for f in (files_output or '').split('\n') if f]
    num_files = len(file_list)
    num_lines = 0
    num_chars = 0
    if not files_result["ok"]:
        diagnostics.append(command_failure_message(files_result))
    if num_files > 0:
        wc_result = run_command(f"git ls-files -- {git_patterns} | xargs wc -lc 2>/dev/null")
        wc_output = command_output(wc_result)
        if wc_output:
            total_line = wc_output.strip().split('\n')[-1]
            parts = total_line.split()
            if len(parts) >= 2:
                num_lines = int(parts[0])
                num_chars = int(parts[1])
        elif not wc_result["ok"]:
            diagnostics.append(command_failure_message(wc_result))
    num_tokens = num_chars // 4  # assume approximately 4 chars per token

    # count dependencies via uv.lock
    uv_lock_lines = 0
    if os.path.exists('uv.lock'):
        with open('uv.lock', 'r', encoding='utf-8') as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    if diagnostics:
        header += "### Diagnostics\n"
        for diagnostic in diagnostics:
            header += f"- {diagnostic}\n"
        header += "\n"
    return header

# -----------------------------------------------------------------------------

def slugify(text):
    """Slugify a text string."""
    return text.lower().replace(" ", "-")

# the expected files and their order
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# the metrics we're currently interested in
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]

def extract(section, keys):
    """simple def to extract a single key from a section"""
    if not isinstance(keys, list):
        keys = [keys] # convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out

def extract_timestamp(content, prefix):
    """Extract timestamp from content with given prefix."""
    for line in content.split('\n'):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                pass
    return None

class Report:
    """Maintains a bunch of logs, generates a final markdown report."""

    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """Log a section of data to the report."""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # directly write the string
                    f.write(item)
                else:
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """Generate the final report."""
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"Generating report to {report_file}")
        final_metrics = {} # the most important final metrics we'll add as table at the end
        start_time = None
        end_time = None
        diagnostics = []
        with open(report_file, "w", encoding="utf-8") as out_file:
            # write the header first
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    if start_time is None:
                        diagnostics.append("Failed to parse start timestamp from header.md")
                    # capture bloat data for summary later (the stuff after Bloat header and until \n\n)
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None # will cause us to not write the total wall clock time
                bloat_data = "[bloat data missing]"
                print(f"Warning: {header_file} does not exist. Did you forget to run `nanochat reset`?")
                diagnostics.append("header.md is missing; report summary is incomplete")
            # process all the individual sections
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    diagnostics.append(f"Missing report section: {file_name}")
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file:
                    section = in_file.read()
                # Extract timestamp from this section (the last section's timestamp will "stick" as end_time)
                if "rl" not in file_name:
                    # Skip RL sections for end_time calculation because RL is experimental
                    end_time = extract_timestamp(section, "timestamp:")
                    if end_time is None:
                        diagnostics.append(f"Failed to parse timestamp from {file_name}")
                # extract the most important metrics from the sections
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K") # RL only evals GSM8K
                # append this section of the report
                out_file.write(section)
                out_file.write("\n")
            if diagnostics:
                out_file.write("## Diagnostics\n\n")
                for diagnostic in diagnostics:
                    out_file.write(f"- {diagnostic}\n")
                out_file.write("\n")
            # add the final metrics table
            out_file.write("## Summary\n\n")
            # Copy over the bloat metrics from the header
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # Collect all unique metric names
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # Custom ordering: CORE first, ChatCORE last, rest in middle
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            # Fixed column widths
            stages = ["base", "sft", "rl"]
            metric_width = 15
            value_width = 8
            # Write table header
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # Write separator
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # Write table rows
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        # also cp the report.md file to current directory
        print(f"Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """Reset the report."""
        # Remove section files
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove report.md if it exists
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # Generate and write the header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

class DummyReport:
    def log(self, *args, **kwargs):
        pass
    def reset(self, *args, **kwargs):
        pass

def get_report():
    # just for convenience, only rank 0 logs to report
    from nanochat.common import get_base_dir, get_dist_info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate or reset nanochat training reports.")
    parser.add_argument("command", nargs="?", default="generate", choices=["generate", "reset"], help="Operation to perform (default: generate)")
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
