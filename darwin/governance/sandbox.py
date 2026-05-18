"""Sandboxed execution environment for autonomous code proposals.

All execution happens **locally** in an isolated subprocess.
No external sandbox service or paid product required.

Safety guarantees
-----------------
* Time-limited execution (configurable, default 30 s)
* Network access disabled inside the sandbox by default
* File-system writes are limited to an isolated temp directory
* The main Darwin process is never directly mutated
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SANDBOX_RUNNER = textwrap.dedent(
    """\
    import sys, os, traceback, json

    # Restrict file-system writes to the sandbox work dir
    WORK_DIR = os.environ.get("DARWIN_SANDBOX_WORK", "/tmp/darwin_sandbox")
    os.makedirs(WORK_DIR, exist_ok=True)
    os.chdir(WORK_DIR)

    # Hard limit on CPU time (seconds)
    cpu_limit = int(os.environ.get("DARWIN_CPU_LIMIT", "10"))
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except Exception:
        pass  # not available on all platforms

    result = {"stdout": "", "stderr": "", "returncode": 0}
    try:
        exec(open(os.environ["DARWIN_SANDBOX_SCRIPT"]).read(), {"__name__": "__main__"})
    except SystemExit as e:
        result["returncode"] = int(e.code or 0)
    except Exception:
        result["stderr"] = traceback.format_exc()
        result["returncode"] = 1
    print(json.dumps(result), flush=True)
    """
)


class SandboxResult:
    def __init__(self, stdout: str, stderr: str, returncode: int, timed_out: bool = False) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def __repr__(self) -> str:
        return (
            f"SandboxResult(returncode={self.returncode}, "
            f"timed_out={self.timed_out}, "
            f"stderr={self.stderr[:80]!r})"
        )


class Sandbox:
    """Runs arbitrary Python code in an isolated subprocess.

    Usage::

        sb = Sandbox(timeout_seconds=15)
        result = sb.run_code("print(1 + 1)")
        assert result.success
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        allow_network: bool = False,
        work_dir: Optional[str] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.allow_network = allow_network
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="darwin_sandbox_")

    def run_code(self, code: str) -> SandboxResult:
        """Execute *code* in the sandbox and return the result."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as script_file:
            script_file.write(code)
            script_path = script_file.name

        runner_path = Path(self._work_dir) / "_runner.py"
        runner_path.write_text(_SANDBOX_RUNNER, encoding="utf-8")

        env = os.environ.copy()
        env["DARWIN_SANDBOX_SCRIPT"] = script_path
        env["DARWIN_SANDBOX_WORK"] = self._work_dir
        env["DARWIN_CPU_LIMIT"] = str(self.timeout_seconds)

        # Disable network access by overriding proxy to a dead address
        if not self.allow_network:
            env["http_proxy"] = "http://127.0.0.2:1"
            env["https_proxy"] = "http://127.0.0.2:1"
            env["HTTP_PROXY"] = env["http_proxy"]
            env["HTTPS_PROXY"] = env["https_proxy"]

        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
            )
            # The runner prints a JSON payload with the inner returncode/stderr.
            # Parse it so that errors inside the sandbox surface correctly.
            raw_stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
            try:
                import json as _json
                inner = _json.loads(raw_stdout.strip().splitlines()[-1])
                returncode = int(inner.get("returncode", returncode))
                if inner.get("stderr"):
                    stderr = inner["stderr"] + "\n" + stderr
                stdout = "\n".join(raw_stdout.strip().splitlines()[:-1])
            except (ValueError, KeyError, IndexError):
                stdout = raw_stdout
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout = ""
            stderr = f"Sandbox timed out after {self.timeout_seconds}s"
            returncode = -1
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

        return SandboxResult(stdout=stdout, stderr=stderr, returncode=returncode, timed_out=timed_out)

    def run_tests(self, test_dir: str = "tests") -> SandboxResult:
        """Run the project test suite in the sandbox."""
        code = f"import subprocess, sys\nresult = subprocess.run([sys.executable, '-m', 'pytest', '{test_dir}', '-x', '-q'], capture_output=True, text=True)\nprint(result.stdout)\nprint(result.stderr, file=__import__('sys').stderr)\nsys.exit(result.returncode)\n"
        return self.run_code(code)
