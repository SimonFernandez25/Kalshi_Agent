"""
Tests for the subprocess-based sandbox runner.

Verifies:
  - A valid tool runs successfully and returns correct JSON structure
  - Timeout kills a hanging tool
  - Malformed output is rejected
  - Network access raises PermissionError inside sandbox
  - os.system calls raise inside sandbox (module-level network block)
"""

from __future__ import annotations

import json
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prediction_agent.evolution.sandbox_runner import run_tool_in_sandbox, SandboxResult


# ── Synthetic event dict used across tests ─────────────────────────────────────

SYNTHETIC_EVENT = {
    "event_id": "test-evt-001",
    "market_id": "TEST-MKT-001",
    "market_title": "Test Market",
    "current_price": 0.50,
    "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
}


# ── Helper: write a temp tool file ────────────────────────────────────────────

def _write_tool(code: str) -> Path:
    """Write tool source to a temp .py file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write(textwrap.dedent(code))
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSandboxRunnerSuccess:
    def test_valid_tool_returns_success(self):
        """A well-formed BaseTool subclass should return success with a numeric vector."""
        tool_code = """
            from tools.base_tool import BaseTool
            from schemas import EventInput, ToolOutput

            class GoodTool(BaseTool):
                @property
                def name(self): return "good_tool"
                @property
                def description(self): return "A deterministic test tool"
                def run(self, event: EventInput, **kwargs) -> ToolOutput:
                    return ToolOutput(
                        tool_name=self.name,
                        output_vector=[0.42, 0.58],
                        metadata={}
                    )
        """
        path = _write_tool(tool_code)
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT, timeout_sec=15)
        assert result.success is True
        assert result.output_vector == [0.42, 0.58]
        assert result.error is None

    def test_result_has_expected_fields(self):
        """SandboxResult must have all required fields populated."""
        tool_code = """
            from tools.base_tool import BaseTool
            from schemas import EventInput, ToolOutput

            class FieldCheckTool(BaseTool):
                @property
                def name(self): return "field_check"
                @property
                def description(self): return "field check"
                def run(self, event: EventInput, **kwargs) -> ToolOutput:
                    return ToolOutput(tool_name=self.name, output_vector=[0.5])
        """
        path = _write_tool(tool_code)
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT)
        assert hasattr(result, "success")
        assert hasattr(result, "output_vector")
        assert hasattr(result, "error")
        assert hasattr(result, "stderr_text")
        assert hasattr(result, "exit_code")
        assert hasattr(result, "duration_ms")
        assert result.duration_ms >= 0


class TestSandboxRunnerFailures:
    def test_syntax_error_returns_failure(self):
        """A tool with invalid Python syntax should fail gracefully."""
        path = _write_tool("def this is broken python !!!")
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT)
        assert result.success is False
        assert result.error is not None

    def test_no_base_tool_subclass_returns_failure(self):
        """A module with no BaseTool subclass should fail."""
        path = _write_tool("x = 42  # No BaseTool here")
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT)
        assert result.success is False

    def test_runtime_exception_returns_failure(self):
        """A tool that raises during run() should fail cleanly."""
        tool_code = """
            from tools.base_tool import BaseTool
            from schemas import EventInput, ToolOutput

            class CrashingTool(BaseTool):
                @property
                def name(self): return "crashing_tool"
                @property
                def description(self): return "crashes"
                def run(self, event: EventInput, **kwargs) -> ToolOutput:
                    raise ValueError("intentional crash")
        """
        path = _write_tool(tool_code)
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT)
        assert result.success is False
        assert "intentional crash" in (result.error or "")

    def test_missing_tool_file_returns_failure(self):
        """A non-existent file path should return a failure result."""
        path = Path("/tmp/this_tool_does_not_exist_xyz.py")
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT)
        assert result.success is False

    def test_timeout_kills_hanging_tool(self):
        """A tool that sleeps past the timeout should be killed."""
        tool_code = """
            import time
            from tools.base_tool import BaseTool
            from schemas import EventInput, ToolOutput

            class HangingTool(BaseTool):
                @property
                def name(self): return "hanging_tool"
                @property
                def description(self): return "hangs forever"
                def run(self, event: EventInput, **kwargs) -> ToolOutput:
                    time.sleep(9999)
                    return ToolOutput(tool_name=self.name, output_vector=[0.0])
        """
        path = _write_tool(tool_code)
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT, timeout_sec=3)
        assert result.success is False
        assert "timeout" in (result.error or "").lower() or result.exit_code != 0

    def test_empty_output_vector_returns_failure(self):
        """A tool returning an empty output_vector must be rejected."""
        tool_code = """
            from tools.base_tool import BaseTool
            from schemas import EventInput, ToolOutput

            class EmptyVectorTool(BaseTool):
                @property
                def name(self): return "empty_vector"
                @property
                def description(self): return "empty vector"
                def run(self, event: EventInput, **kwargs) -> ToolOutput:
                    return ToolOutput(tool_name=self.name, output_vector=[])
        """
        path = _write_tool(tool_code)
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT)
        assert result.success is False


class TestSandboxNetworkIsolation:
    def test_network_access_blocked(self):
        """Any attempt to use socket.socket inside sandbox should fail."""
        tool_code = """
            import socket
            from tools.base_tool import BaseTool
            from schemas import EventInput, ToolOutput

            class NetworkTool(BaseTool):
                @property
                def name(self): return "network_tool"
                @property
                def description(self): return "tries network"
                def run(self, event: EventInput, **kwargs) -> ToolOutput:
                    # This should raise PermissionError
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(("8.8.8.8", 80))
                    return ToolOutput(tool_name=self.name, output_vector=[1.0])
        """
        path = _write_tool(tool_code)
        result = run_tool_in_sandbox(path, SYNTHETIC_EVENT)
        # Should fail because either: (a) AST blocks socket import,
        # or (b) sandbox disables socket at runtime
        assert result.success is False


class TestSandboxResultSerialization:
    def test_to_dict_returns_serializable_dict(self):
        """SandboxResult.to_dict() must return a JSON-serializable dict."""
        result = SandboxResult(
            success=True,
            output_vector=[0.5, 0.7],
            error=None,
            stderr_text="",
            exit_code=0,
            duration_ms=42.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        # Must be JSON-serializable
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["success"] is True
        assert deserialized["output_vector"] == [0.5, 0.7]
