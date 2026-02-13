"""
Tests for tool_verifier.

Verifies:
  - Safe tool passes all checks
  - AST blocks disallowed imports (os, subprocess, random)
  - AST blocks exec/eval calls
  - Runtime check catches non-ToolOutput returns
  - Determinism check catches non-deterministic tools
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from prediction_agent.evolution.schemas import ToolSpec
from prediction_agent.evolution.tool_verifier import verify_tool


def _default_spec(name: str = "test_tool") -> ToolSpec:
    return ToolSpec(
        tool_name=name,
        description="Test tool",
        inputs={},
        output_type="float",
        deterministic=True,
        data_sources=[],
        expected_token_reduction=0.0,
        expected_accuracy_gain=0.0,
        risk_level="low",
    )


class TestASTInspection:

    def test_safe_tool_passes(self, tmp_path: Path):
        code = dedent('''\
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class SafeTool(BaseTool):
                @property
                def name(self) -> str:
                    return "safe_tool"

                @property
                def description(self) -> str:
                    return "A safe test tool."

                def run(self, event: EventInput, **kwargs) -> ToolOutput:
                    return ToolOutput(
                        tool_name=self.name,
                        output_vector=[event.current_price * 0.5],
                        metadata={},
                    )
        ''')
        tool_file = tmp_path / "safe_tool.py"
        tool_file.write_text(code, encoding="utf-8")

        result = verify_tool(tool_file, _default_spec("safe_tool"))
        assert result.passed is True
        assert result.checks["ast_inspection"] is True
        assert result.checks["runtime_sandbox"] is True
        assert result.checks["determinism"] is True

    def test_blocks_os_import(self, tmp_path: Path):
        code = dedent('''\
            import os
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class BadTool(BaseTool):
                @property
                def name(self): return "bad_tool"
                @property
                def description(self): return "Bad."
                def run(self, event, **kwargs):
                    return ToolOutput(tool_name="bad_tool", output_vector=[0.0])
        ''')
        tool_file = tmp_path / "bad_tool.py"
        tool_file.write_text(code, encoding="utf-8")

        result = verify_tool(tool_file, _default_spec("bad_tool"))
        assert result.passed is False
        assert "os" in result.rejection_reason

    def test_blocks_subprocess_import(self, tmp_path: Path):
        code = dedent('''\
            import subprocess
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class BadTool(BaseTool):
                @property
                def name(self): return "bad_tool"
                @property
                def description(self): return "Bad."
                def run(self, event, **kwargs):
                    return ToolOutput(tool_name="bad_tool", output_vector=[0.0])
        ''')
        tool_file = tmp_path / "bad_tool.py"
        tool_file.write_text(code, encoding="utf-8")

        result = verify_tool(tool_file, _default_spec("bad_tool"))
        assert result.passed is False
        assert "subprocess" in result.rejection_reason

    def test_blocks_random_import(self, tmp_path: Path):
        code = dedent('''\
            import random
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class BadTool(BaseTool):
                @property
                def name(self): return "bad_tool"
                @property
                def description(self): return "Bad."
                def run(self, event, **kwargs):
                    return ToolOutput(tool_name="bad_tool", output_vector=[random.random()])
        ''')
        tool_file = tmp_path / "bad_tool.py"
        tool_file.write_text(code, encoding="utf-8")

        result = verify_tool(tool_file, _default_spec("bad_tool"))
        assert result.passed is False
        assert "random" in result.rejection_reason.lower()

    def test_blocks_eval_call(self, tmp_path: Path):
        code = dedent('''\
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class BadTool(BaseTool):
                @property
                def name(self): return "bad_tool"
                @property
                def description(self): return "Bad."
                def run(self, event, **kwargs):
                    val = eval("1 + 2")
                    return ToolOutput(tool_name="bad_tool", output_vector=[float(val)])
        ''')
        tool_file = tmp_path / "bad_tool.py"
        tool_file.write_text(code, encoding="utf-8")

        result = verify_tool(tool_file, _default_spec("bad_tool"))
        assert result.passed is False
        assert "eval" in result.rejection_reason


class TestRuntimeSandbox:

    def test_catches_non_tool_output(self, tmp_path: Path):
        code = dedent('''\
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class BadReturnTool(BaseTool):
                @property
                def name(self): return "bad_return_tool"
                @property
                def description(self): return "Returns wrong type."
                def run(self, event, **kwargs):
                    return "not a ToolOutput"
        ''')
        tool_file = tmp_path / "bad_return_tool.py"
        tool_file.write_text(code, encoding="utf-8")

        result = verify_tool(tool_file, _default_spec("bad_return_tool"))
        assert result.passed is False
        assert result.checks.get("ast_inspection") is True
        assert result.checks.get("runtime_sandbox") is False

    def test_catches_empty_output_vector(self, tmp_path: Path):
        code = dedent('''\
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class EmptyVecTool(BaseTool):
                @property
                def name(self): return "empty_vec_tool"
                @property
                def description(self): return "Empty vector."
                def run(self, event, **kwargs):
                    return ToolOutput(tool_name="empty_vec_tool", output_vector=[])
        ''')
        tool_file = tmp_path / "empty_vec_tool.py"
        tool_file.write_text(code, encoding="utf-8")

        # This should fail at runtime because ToolOutput requires min_length=1
        result = verify_tool(tool_file, _default_spec("empty_vec_tool"))
        assert result.passed is False


class TestDeterminism:

    def test_deterministic_tool_passes(self, tmp_path: Path):
        code = dedent('''\
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class DetTool(BaseTool):
                @property
                def name(self): return "det_tool"
                @property
                def description(self): return "Deterministic."
                def run(self, event, **kwargs):
                    return ToolOutput(
                        tool_name="det_tool",
                        output_vector=[event.current_price * 2.0],
                    )
        ''')
        tool_file = tmp_path / "det_tool.py"
        tool_file.write_text(code, encoding="utf-8")

        result = verify_tool(tool_file, _default_spec("det_tool"))
        assert result.passed is True
        assert result.checks["determinism"] is True
