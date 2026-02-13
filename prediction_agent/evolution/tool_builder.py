"""
Tool builder -- converts a ToolSpec into a Python source file.

Calls the LLM to generate code that subclasses BaseTool.
Output is written to tools/generated/<tool_name>.py.
Does NOT register the tool -- that happens only after verification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import GENERATED_TOOLS_DIR, LLM_TEMPERATURE
from prediction_agent.evolution.schemas import ToolSpec

logger = logging.getLogger(__name__)

_BUILD_PROMPT_TEMPLATE = """You are a Python code generator for a prediction agent system.

Generate a complete Python module that implements a new deterministic tool.

The tool must:
1. Subclass BaseTool from tools.base_tool
2. Implement these abstract properties and method:
   - name (property) -> str: return "{tool_name}"
   - description (property) -> str: return "{description}"
   - run(self, event: EventInput, **kwargs) -> ToolOutput

3. Accept EventInput from schemas (has: event_id, market_id, market_title, current_price, timestamp)
4. Return ToolOutput from schemas (has: tool_name, output_vector (List[float]), metadata (dict))
5. Be fully deterministic: same input must always produce same output
6. Include __version__ = "0.1.0" at module level
7. Include a _self_test() function that creates a test EventInput, runs the tool, and asserts the output is valid

Tool specification:
- Name: {tool_name}
- Description: {description}
- Inputs: {inputs}
- Output type: {output_type}
- Data sources: {data_sources}

Allowed imports: json, logging, statistics, math, pathlib, datetime, typing, collections, re
Allowed imports from project: schemas, tools.base_tool, config

FORBIDDEN:
- import os, subprocess, sys, shutil, socket, http, requests, urllib
- os.system(), subprocess.run(), exec(), eval()
- open() in write mode
- Any network calls outside of reading local files
- random module (must be deterministic)

The tool should read from local JSONL snapshot files if it needs data.
Default snapshot path: Path(__file__).resolve().parent.parent / "outputs" / "market_snapshots.jsonl"

Output ONLY the Python code. No markdown fences. No explanation.
Start directly with the docstring or imports."""


def build_tool(spec: ToolSpec, output_dir: Path | None = None) -> Path:
    """
    Generate a Python tool file from a ToolSpec.

    Args:
        spec: The validated tool specification.
        output_dir: Directory to write the tool file. Defaults to GENERATED_TOOLS_DIR.

    Returns:
        Path to the generated Python file.

    Raises:
        ValueError: If the LLM response is empty or unusable.
    """
    from langchain_aws import ChatBedrock
    from langchain_core.messages import HumanMessage

    target_dir = output_dir or GENERATED_TOOLS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    llm = ChatBedrock(
        model_id="arn:aws:bedrock:us-east-1:770182232673:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        provider="anthropic",
        region_name="us-east-1",
        temperature=LLM_TEMPERATURE,
    )

    prompt = _BUILD_PROMPT_TEMPLATE.format(
        tool_name=spec.tool_name,
        description=spec.description,
        inputs=json.dumps(spec.inputs),
        output_type=spec.output_type,
        data_sources=json.dumps(spec.data_sources),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content.strip()

    # Strip markdown fences if present
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
        if code.endswith("```"):
            code = code[: code.rfind("```")]
        code = code.strip()

    if not code or len(code) < 50:
        raise ValueError("LLM produced empty or trivially short tool code.")

    # Write to file
    file_path = target_dir / f"{spec.tool_name}.py"
    file_path.write_text(code, encoding="utf-8")

    logger.info("Built tool file: %s (%d bytes)", file_path, len(code))
    return file_path
