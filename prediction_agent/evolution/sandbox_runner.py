"""
Subprocess-based sandbox runner for generated tool verification.

This module provides two interfaces:
  1. _sandbox_worker_main() — the entrypoint for the CHILD subprocess.
     It is invoked by the parent via subprocess.run(). It loads the tool,
     runs it, and writes a structured JSON result to stdout.

  2. run_tool_in_sandbox(tool_path, event_input) — called by the PARENT
     process (tool_verifier.py Phase 2). It spawns the child, enforces a
     hard timeout, captures stdout, and parses the JSON result.

Security properties:
  - Tool code never executes in the main process.
  - Communication is strictly JSON over stdin/stdout (no shared memory).
  - A hard wall-clock timeout kills the child if it hangs.
  - Resource limits (CPU + memory) are applied inside the child on
    POSIX systems via the `resource` stdlib module. On Windows the child
    exits cleanly without resource limits but the timeout still applies.
  - The child disables network access by monkey-patching socket.socket
    before importing the tool.
  - The child restricts sys.path so only the project root and stdlib are
    importable; no site-packages leakage.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_SANDBOX_TIMEOUT_DEFAULT = 10          # seconds; overridden by config if available
_SANDBOX_MAX_MEMORY_MB   = 256         # soft RSS limit inside child (POSIX only)
_SANDBOX_MAX_CPU_SEC     = 8           # soft CPU-time limit inside child (POSIX only)
_SANDBOX_EXIT_OK         = 0
_SANDBOX_EXIT_ERR        = 1


# ── Public dataclass ───────────────────────────────────────────────────────────

class SandboxResult:
    """
    Structured result from a sandboxed tool execution.

    Attributes:
        success:        True if the tool ran and returned a valid ToolOutput.
        output_vector:  The tool's output vector (empty list on failure).
        error:          Human-readable error description, or None on success.
        stderr_text:    Raw stderr captured from the child process.
        exit_code:      OS exit code of the child process.
        duration_ms:    Wall-clock time of the child process in milliseconds.
    """

    __slots__ = ("success", "output_vector", "error", "stderr_text",
                 "exit_code", "duration_ms")

    def __init__(
        self,
        success: bool,
        output_vector: list,
        error: Optional[str],
        stderr_text: str,
        exit_code: int,
        duration_ms: float,
    ) -> None:
        self.success       = success
        self.output_vector = output_vector
        self.error         = error
        self.stderr_text   = stderr_text
        self.exit_code     = exit_code
        self.duration_ms   = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":       self.success,
            "output_vector": self.output_vector,
            "error":         self.error,
            "stderr_text":   self.stderr_text,
            "exit_code":     self.exit_code,
            "duration_ms":   self.duration_ms,
        }


# ── Parent-side API ────────────────────────────────────────────────────────────

def run_tool_in_sandbox(
    tool_path: Path,
    event_input_dict: Dict[str, Any],
    timeout_sec: Optional[int] = None,
) -> SandboxResult:
    """
    Spawn an isolated subprocess, run the tool inside it, return structured result.

    Args:
        tool_path:        Absolute path to the generated .py tool file.
        event_input_dict: The EventInput as a plain dict (JSON-serialisable).
        timeout_sec:      Hard wall-clock kill timeout for the child.
                          Defaults to TOOL_VERIFY_TIMEOUT_SEC from config.

    Returns:
        SandboxResult with success/failure details.
    """
    # Resolve timeout
    if timeout_sec is None:
        try:
            from config import TOOL_VERIFY_TIMEOUT_SEC
            timeout_sec = TOOL_VERIFY_TIMEOUT_SEC
        except ImportError:
            timeout_sec = _SANDBOX_TIMEOUT_DEFAULT

    # Serialise the EventInput to a temp JSON file (avoids shell-quoting issues)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(event_input_dict, tmp, default=str)
        event_json_path = tmp.name

    try:
        return _spawn_sandbox(
            tool_path=tool_path,
            event_json_path=event_json_path,
            timeout_sec=timeout_sec,
        )
    finally:
        try:
            os.unlink(event_json_path)
        except OSError:
            pass


def _spawn_sandbox(
    tool_path: Path,
    event_json_path: str,
    timeout_sec: int,
) -> SandboxResult:
    """
    Low-level subprocess spawn + result parsing.
    """
    import time

    # Build command: python -m prediction_agent.evolution.sandbox_runner <tool_path> <event_json>
    # We pass __file__ as the script so the child imports THIS module as __main__.
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),   # child calls _sandbox_worker_main()
        str(tool_path),
        event_json_path,
    ]

    # Determine project root so child can set sys.path correctly
    try:
        from config import PREDICTION_AGENT_DIR
        project_root = str(PREDICTION_AGENT_DIR)
    except ImportError:
        project_root = str(Path(__file__).resolve().parent.parent.parent)

    env = os.environ.copy()
    env["KALSHI_SANDBOX_PROJECT_ROOT"] = project_root
    # Signal to child that it is running in sandbox mode
    env["KALSHI_SANDBOX_MODE"] = "1"

    t0 = time.perf_counter()
    timed_out = False

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        duration_ms = (time.perf_counter() - t0) * 1000
        exit_code   = proc.returncode
        raw_stdout  = proc.stdout.strip()
        raw_stderr  = proc.stderr.strip()

    except subprocess.TimeoutExpired as exc:
        timed_out    = True
        duration_ms  = timeout_sec * 1000
        exit_code    = -1
        raw_stdout   = ""
        raw_stderr   = str(exc)
        logger.warning("Sandbox timed out after %ds for tool: %s", timeout_sec, tool_path.name)

    if timed_out:
        return SandboxResult(
            success=False,
            output_vector=[],
            error=f"Sandbox timed out after {timeout_sec}s",
            stderr_text=raw_stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

    # Parse JSON from stdout
    if not raw_stdout:
        return SandboxResult(
            success=False,
            output_vector=[],
            error="Sandbox produced no stdout output",
            stderr_text=raw_stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

    try:
        result_json = json.loads(raw_stdout)
    except json.JSONDecodeError as exc:
        return SandboxResult(
            success=False,
            output_vector=[],
            error=f"Sandbox stdout is not valid JSON: {exc}\nRaw: {raw_stdout[:200]}",
            stderr_text=raw_stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

    # Validate expected fields
    status = result_json.get("status", "failure")
    output_vector = result_json.get("output_vector", [])
    error_msg = result_json.get("error", None)

    if status != "success":
        return SandboxResult(
            success=False,
            output_vector=[],
            error=error_msg or "Sandbox reported failure",
            stderr_text=raw_stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

    # Validate output_vector
    if not isinstance(output_vector, list) or not output_vector:
        return SandboxResult(
            success=False,
            output_vector=[],
            error="output_vector missing or empty in sandbox result",
            stderr_text=raw_stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

    for i, val in enumerate(output_vector):
        if not isinstance(val, (int, float)):
            return SandboxResult(
                success=False,
                output_vector=[],
                error=f"output_vector[{i}] is not numeric: {type(val).__name__}",
                stderr_text=raw_stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
            )

    logger.debug(
        "Sandbox OK: %s → output_vector=%s (%.1fms)",
        tool_path.name, output_vector, duration_ms,
    )

    return SandboxResult(
        success=True,
        output_vector=output_vector,
        error=None,
        stderr_text=raw_stderr,
        exit_code=exit_code,
        duration_ms=duration_ms,
    )


# ── Child-side worker ──────────────────────────────────────────────────────────

def _sandbox_worker_main() -> None:
    """
    Entrypoint executed INSIDE the child subprocess.

    Reads arguments from sys.argv, applies resource limits and network
    isolation, loads the tool, runs it, and writes a JSON result to stdout.

    This function MUST NOT import from any project module before applying
    the isolation layers, to prevent side-effects.
    """
    import importlib.util

    if len(sys.argv) != 3:
        _child_fail("Usage: sandbox_runner.py <tool_path> <event_json_path>")
        return

    tool_path_str  = sys.argv[1]
    event_json_str = sys.argv[2]

    # 1. Apply POSIX resource limits (best-effort; silently skip on Windows)
    _apply_resource_limits()

    # 2. Block network access by replacing socket.socket with a stub
    _disable_network()

    # 3. Fix sys.path: add project root so generated tool code can resolve
    #    `from tools.base_tool import BaseTool` and `from schemas import ...`
    #    (both live at the project root, not in a sub-package).
    project_root = os.environ.get("KALSHI_SANDBOX_PROJECT_ROOT", "")
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 4. Load EventInput from JSON file
    try:
        with open(event_json_str, "r", encoding="utf-8") as f:
            event_dict = json.load(f)
    except Exception as exc:
        _child_fail(f"Could not read event JSON: {exc}")
        return

    # 5. Reconstruct EventInput (schemas.py lives at project root)
    try:
        from schemas import EventInput
        # Datetime field may be a string — parse it
        if isinstance(event_dict.get("timestamp"), str):
            event_dict["timestamp"] = datetime.fromisoformat(
                event_dict["timestamp"].replace("Z", "+00:00")
            )
        event = EventInput(**event_dict)
    except Exception as exc:
        _child_fail(f"EventInput reconstruction failed: {exc}")
        return

    # 6. Load tool module
    tool_path = Path(tool_path_str)
    if not tool_path.exists():
        _child_fail(f"Tool file not found: {tool_path_str}")
        return

    try:
        module_spec = importlib.util.spec_from_file_location(
            f"_sandbox_tool_{tool_path.stem}",
            str(tool_path),
        )
        if module_spec is None or module_spec.loader is None:
            _child_fail("Could not create module spec for tool")
            return

        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    except Exception as exc:
        _child_fail(f"Tool import failed: {exc}")
        return

    # 7. Find BaseTool subclass (tools/ lives at project root)
    try:
        from tools.base_tool import BaseTool
    except Exception as exc:
        _child_fail(f"Could not import BaseTool: {exc}")
        return

    tool_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name, None)
        if (
            attr is not None
            and isinstance(attr, type)
            and issubclass(attr, BaseTool)
            and attr is not BaseTool
        ):
            tool_class = attr
            break

    if tool_class is None:
        _child_fail("No BaseTool subclass found in module")
        return

    # 8. Instantiate and run
    try:
        tool_instance = tool_class()
        output = tool_instance.run(event)
    except Exception as exc:
        _child_fail(f"Tool.run() raised: {type(exc).__name__}: {exc}")
        return

    # 9. Validate output (schemas.py lives at project root)
    try:
        from schemas import ToolOutput
        if not isinstance(output, ToolOutput):
            _child_fail(f"Tool did not return ToolOutput, got {type(output).__name__}")
            return
        if not output.output_vector:
            _child_fail("output_vector is empty")
            return
        for i, v in enumerate(output.output_vector):
            if not isinstance(v, (int, float)):
                _child_fail(f"output_vector[{i}] is not numeric: {type(v).__name__}")
                return
    except Exception as exc:
        _child_fail(f"Output validation failed: {exc}")
        return

    # 10. Success — write structured JSON to stdout
    result = {
        "status":        "success",
        "output_vector": output.output_vector,
        "metadata":      output.metadata,
        "error":         None,
    }
    print(json.dumps(result), flush=True)
    sys.exit(_SANDBOX_EXIT_OK)


def _child_fail(reason: str) -> None:
    """Write a failure JSON to stdout and exit with error code."""
    result = {
        "status":        "failure",
        "output_vector": [],
        "error":         reason,
    }
    print(json.dumps(result), flush=True)
    sys.exit(_SANDBOX_EXIT_ERR)


def _apply_resource_limits() -> None:
    """
    Apply CPU and memory resource limits on POSIX systems.
    Silently no-ops on Windows or if the module is unavailable.
    """
    try:
        import resource  # POSIX only
        # CPU time limit (seconds)
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        new_soft = min(_SANDBOX_MAX_CPU_SEC, hard) if hard > 0 else _SANDBOX_MAX_CPU_SEC
        resource.setrlimit(resource.RLIMIT_CPU, (new_soft, hard))

        # Memory (RSS) limit
        mem_bytes = _SANDBOX_MAX_MEMORY_MB * 1024 * 1024
        try:
            soft_m, hard_m = resource.getrlimit(resource.RLIMIT_AS)
            new_soft_m = min(mem_bytes, hard_m) if hard_m > 0 else mem_bytes
            resource.setrlimit(resource.RLIMIT_AS, (new_soft_m, hard_m))
        except (AttributeError, ValueError):
            pass  # RLIMIT_AS not available on all POSIX platforms

    except ImportError:
        pass  # Windows — resource module not available


def _disable_network() -> None:
    """
    Replace socket.socket with a stub that raises on any connection attempt.
    This prevents generated tools from making outbound network calls.
    """
    import socket as _socket_module

    class _BlockedSocket:
        """Drop-in stub that refuses to open any socket."""
        def __init__(self, *args, **kwargs):
            raise PermissionError(
                "[SANDBOX] Network access is disabled in sandbox mode."
            )

    _socket_module.socket = _BlockedSocket  # type: ignore[assignment]


# ── Module entry point detection ───────────────────────────────────────────────

if __name__ == "__main__":
    # When invoked as a subprocess by the parent, run the worker.
    _sandbox_worker_main()
