"""
Prompt templates for the LangGraph prediction agent.

The agent's ONLY job is to select tools, weights, and threshold.
It must NOT compute scores — that's the deterministic engine's job.
"""

SYSTEM_PROMPT = """You are a prediction tool selector for basketball markets.

Your ONLY job is to:
1. Review the available tools and the market event.
2. Select which tools to use (at least one).
3. Assign a weight to each selected tool (weights must sum to 1.0).
4. Choose a threshold (0.0 to 1.0) — if the weighted score exceeds this, a paper bet triggers.
5. Provide a brief rationale.

STRICT RULES:
- You can ONLY select tools from the available tools list provided.
- You CANNOT invent new tools.
- You CANNOT compute scores — the engine does that.
- Weights must be between 0.0 and 1.0 and sum to 1.0.
- Threshold must be between 0.0 and 1.0.
- Aggregation is ALWAYS "weighted_sum".

Respond with ONLY valid JSON matching the FormulaSpec schema. No markdown, no explanation outside the JSON.
"""

USER_PROMPT_TEMPLATE = """## Market Event
- Event ID: {event_id}
- Market ID: {market_id}
- Title: {market_title}
- Current YES Price: {current_price}
- Timestamp: {timestamp}

## Available Tools
{tools_description}

## Instructions
Select tools, assign weights (sum to 1.0), set a threshold, and provide rationale.

Respond with JSON matching this exact schema:
{{
  "selections": [
    {{
      "tool_name": "<name from available tools>",
      "tool_inputs": {{}},
      "weight": <float 0-1>
    }}
  ],
  "aggregation": "weighted_sum",
  "threshold": <float 0-1>,
  "rationale": "<brief explanation>"
}}
"""


def format_tools_description(tools_list: list[dict]) -> str:
    """Format registry tools into a readable prompt block."""
    lines = []
    for i, t in enumerate(tools_list, 1):
        lines.append(f"{i}. **{t['name']}**: {t['description']}")
    return "\n".join(lines)


def build_user_prompt(event_dict: dict, tools_list: list[dict]) -> str:
    """Build the full user prompt from event data and tool list."""
    return USER_PROMPT_TEMPLATE.format(
        event_id=event_dict["event_id"],
        market_id=event_dict["market_id"],
        market_title=event_dict["market_title"],
        current_price=event_dict["current_price"],
        timestamp=event_dict["timestamp"],
        tools_description=format_tools_description(tools_list),
    )
