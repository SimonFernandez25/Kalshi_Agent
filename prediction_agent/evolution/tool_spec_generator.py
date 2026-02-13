"""
Tool spec generator -- LLM-powered structured ToolSpec creation.

Given a GapReport from the analyzer, calls the LLM to produce a
well-formed ToolSpec that can be passed to the tool builder.
"""

from __future__ import annotations

import json
import logging

from config import LLM_MODEL, LLM_TEMPERATURE
from prediction_agent.evolution.schemas import GapReport, ToolSpec

logger = logging.getLogger(__name__)

_SPEC_PROMPT_TEMPLATE = """You are a tool design expert for a prediction agent system.

The system uses deterministic tools that analyze Kalshi prediction market data.
Each tool subclasses BaseTool and implements run(event: EventInput, **kwargs) -> ToolOutput.
Tools return a ToolOutput containing an output_vector of floats.

A capability gap has been detected in the system:

Problem: {problem_detected}
Evidence: {evidence}
Estimated token waste: {estimated_token_waste}

Design a NEW deterministic tool to address this gap.

Respond with ONLY a JSON object matching this exact schema:
{{
    "tool_name": "snake_case_name",
    "description": "One-line description of what the tool does",
    "inputs": {{"param_name": "param_type_description"}},
    "output_type": "float",
    "deterministic": true,
    "data_sources": ["kalshi_snapshots"],
    "expected_token_reduction": 0.0,
    "expected_accuracy_gain": 0.0,
    "risk_level": "low"
}}

Constraints:
- tool_name must be snake_case, descriptive, end with _tool
- Must rely only on Kalshi data or local snapshot data
- Must produce deterministic output (same input -> same output)
- Must not require file IO beyond reading existing snapshots
- Must not modify system state
- risk_level must be "low" or "medium" (never "high")
- deterministic must be true

JSON only. No markdown fences. No explanation."""


def generate_tool_spec(gap: GapReport) -> ToolSpec:
    """
    Call the LLM to produce a structured ToolSpec for the given gap.

    Args:
        gap: The GapReport describing the detected capability gap.

    Returns:
        A validated ToolSpec ready for the tool builder.

    Raises:
        ValueError: If the LLM response cannot be parsed into a valid ToolSpec.
    """
    from langchain_aws import ChatBedrock
    from langchain_core.messages import HumanMessage

    llm = ChatBedrock(
        model_id="arn:aws:bedrock:us-east-1:770182232673:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        provider="anthropic",
        region_name="us-east-1",
        temperature=LLM_TEMPERATURE,
    )

    prompt = _SPEC_PROMPT_TEMPLATE.format(
        problem_detected=gap.problem_detected,
        evidence=json.dumps(gap.evidence, indent=2),
        estimated_token_waste=gap.estimated_token_waste,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_text = response.content.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text[: raw_text.rfind("```")]
        raw_text = raw_text.strip()

    try:
        spec_dict = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for tool spec: {exc}") from exc

    spec = ToolSpec(**spec_dict)

    logger.info(
        "Generated tool spec: name=%s risk=%s",
        spec.tool_name,
        spec.risk_level.value,
    )
    return spec
