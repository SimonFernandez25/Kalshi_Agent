"""
Tests for registry generated tool support.

Verifies:
  - register_generated_tool adds tool to registry
  - Duplicate name raises ValueError
  - load_approved_generated_tools with empty manifest loads nothing
  - load_approved_generated_tools with valid entry loads the tool
  - generated_tool_names property works
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from prediction_agent.schemas import EventInput, ToolOutput
from prediction_agent.tools.base_tool import BaseTool
from prediction_agent.tools.registry import ToolRegistry


class _FakeGeneratedTool(BaseTool):
    """A minimal tool for testing registration."""

    @property
    def name(self) -> str:
        return "fake_generated_tool"

    @property
    def description(self) -> str:
        return "Fake tool for testing."

    def run(self, event: EventInput, **kwargs) -> ToolOutput:
        return ToolOutput(
            tool_name=self.name,
            output_vector=[0.5],
        )


class _AnotherFakeTool(BaseTool):

    @property
    def name(self) -> str:
        return "another_fake_tool"

    @property
    def description(self) -> str:
        return "Another fake."

    def run(self, event: EventInput, **kwargs) -> ToolOutput:
        return ToolOutput(tool_name=self.name, output_vector=[0.3])


class TestRegistryGeneratedTools:

    def test_register_generated_tool(self):
        registry = ToolRegistry()
        tool = _FakeGeneratedTool()
        registry.register_generated_tool(tool, version="0.1.0")

        assert "fake_generated_tool" in registry
        assert registry.get("fake_generated_tool") is tool
        assert "fake_generated_tool" in registry.generated_tool_names

    def test_duplicate_name_raises(self):
        registry = ToolRegistry()
        registry.register_generated_tool(_FakeGeneratedTool())

        with pytest.raises(ValueError, match="conflicts"):
            registry.register_generated_tool(_FakeGeneratedTool())

    def test_empty_manifest_loads_nothing(self, tmp_path: Path, monkeypatch):
        # Create empty approved.json
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        manifest = gen_dir / "approved.json"
        manifest.write_text("[]", encoding="utf-8")

        monkeypatch.setattr(
            "prediction_agent.tools.registry.GENERATED_TOOLS_DIR",
            gen_dir,
            raising=False,
        )

        # Patch the import inside the method
        import prediction_agent.config as cfg
        monkeypatch.setattr(cfg, "GENERATED_TOOLS_DIR", gen_dir)

        registry = ToolRegistry()
        registry.load_approved_generated_tools()

        assert len(registry) == 0

    def test_valid_manifest_loads_tool(self, tmp_path: Path, monkeypatch):
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        # Write a valid tool module
        tool_code = dedent('''\
            from prediction_agent.tools.base_tool import BaseTool
            from prediction_agent.schemas import EventInput, ToolOutput

            class MyGenTool(BaseTool):
                @property
                def name(self) -> str:
                    return "my_gen_tool"

                @property
                def description(self) -> str:
                    return "Generated tool."

                def run(self, event: EventInput, **kwargs) -> ToolOutput:
                    return ToolOutput(
                        tool_name=self.name,
                        output_vector=[event.current_price],
                    )
        ''')
        (gen_dir / "my_gen_tool.py").write_text(tool_code, encoding="utf-8")

        # Write manifest
        manifest = gen_dir / "approved.json"
        manifest.write_text(
            json.dumps([{"tool_name": "my_gen_tool", "version": "0.1.0"}]),
            encoding="utf-8",
        )

        import prediction_agent.config as cfg
        monkeypatch.setattr(cfg, "GENERATED_TOOLS_DIR", gen_dir)

        registry = ToolRegistry()
        registry.load_approved_generated_tools()

        assert "my_gen_tool" in registry
        assert "my_gen_tool" in registry.generated_tool_names

    def test_missing_tool_file_skipped(self, tmp_path: Path, monkeypatch):
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        manifest = gen_dir / "approved.json"
        manifest.write_text(
            json.dumps([{"tool_name": "missing_tool", "version": "0.1.0"}]),
            encoding="utf-8",
        )

        import prediction_agent.config as cfg
        monkeypatch.setattr(cfg, "GENERATED_TOOLS_DIR", gen_dir)

        registry = ToolRegistry()
        registry.load_approved_generated_tools()

        assert "missing_tool" not in registry
        assert len(registry) == 0
