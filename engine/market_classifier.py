"""
Market domain classifier.
=========================
Classifies a Kalshi market into a domain so the agent can select
only relevant tools. Pure string-matching — no LLM, no API calls.

Domains:
  sports     — NBA, NFL, MLB, NHL, college sports, player props
  economics  — CPI, Fed rates, GDP, employment, earnings
  politics   — elections, legislation, approval ratings
  weather    — precipitation, temperature, climate events
  crypto     — BTC, ETH, crypto prices
  other      — anything that doesn't match the above

Tool domain map (which tools are relevant per domain):
  sports:    sportsbook_implied_probability_tool, snapshot_volatility_tool,
             spread_compression_tool, price_jump_detector_tool, liquidity_spike_tool,
             weather_probability_tool (outdoor venues only)
  economics: fred_macro_tool, bls_labor_tool, snapshot_volatility_tool,
             spread_compression_tool, price_jump_detector_tool
  politics:  fred_macro_tool, snapshot_volatility_tool, price_jump_detector_tool,
             liquidity_spike_tool
  weather:   weather_probability_tool, snapshot_volatility_tool
  crypto:    snapshot_volatility_tool, spread_compression_tool, price_jump_detector_tool,
             liquidity_spike_tool
  other:     snapshot_volatility_tool, spread_compression_tool, price_jump_detector_tool
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ── Keyword patterns per domain ────────────────────────────────────────────────

_SPORTS_KEYWORDS = [
    # Pro leagues (unambiguous abbreviations)
    r"\bnba\b", r"\bnfl\b", r"\bmlb\b", r"\bnhl\b",
    r"\bncaa\b", r"\bncaab\b", r"\bncaaf\b",
    # Sport-specific events (require sports context)
    r"\bplayoff\b", r"\bplayoffs\b", r"\bsuperbowl\b", r"\bsuper bowl\b",
    r"\bworld series\b", r"\bstanley cup\b", r"\bnba finals\b", r"\bnfl season\b",
    # Sport-specific terms (not generic English)
    r"\bmoneyline\b", r"\bspread bet\b", r"\bpoint spread\b",
    r"\bassists\b", r"\brebounds\b", r"\btouchdown\b", r"\bhome run\b",
    r"\bfield goal\b", r"\bthree-pointer\b", r"\bprop bet\b",
    # Team names (specific enough to be unambiguous)
    r"\blakers\b", r"\bceltics\b", r"\bwarriors\b", r"\bbucks\b",
    r"\bnuggets\b", r"\bknicks\b", r"\bnets\b", r"\bsuns\b",
    r"\bpackers\b", r"\bchiefs\b", r"\bcowboys\b", r"\bpatriots\b",
    r"\byankees\b", r"\bred sox\b", r"\bdodgers\b", r"\bcubs\b",
    # Basketball-specific player props
    r"\bdouble-double\b", r"\btriple-double\b",
]

_ECONOMICS_KEYWORDS = [
    r"\bcpi\b", r"\binflation\b", r"\bfed\b", r"\bfederal reserve\b",
    r"\binterest rate\b", r"\bgdp\b", r"\brecession\b", r"\bunemployment\b",
    r"\bjobs\b", r"\bnonfarm\b", r"\bpayroll\b", r"\bearnings\b", r"\bpce\b",
    r"\bfomc\b", r"\brate hike\b", r"\brate cut\b", r"\byield\b", r"\btreasury\b",
    r"\bvix\b", r"\bvolatility index\b", r"\bs&p\b", r"\bspx\b", r"\bdow\b",
    r"\bhousing\b", r"\bretail sales\b", r"\btrade deficit\b",
]

_POLITICS_KEYWORDS = [
    r"\belection\b", r"\bvote\b", r"\bpresident\b", r"\bsenate\b", r"\bhouse\b",
    r"\bcongress\b", r"\blegislation\b", r"\bbill\b", r"\bapproval\b",
    r"\bpoll\b", r"\bprimary\b", r"\bcandidate\b", r"\bparty\b",
    r"\bdemocrat\b", r"\brepublican\b", r"\bindependent\b",
    r"\bimpeach\b", r"\bsupreme court\b", r"\bregulation\b",
]

_WEATHER_KEYWORDS = [
    r"\brain\b", r"\bsnow\b", r"\bhurricane\b", r"\btornado\b", r"\bflood\b",
    r"\btemperature\b", r"\bstorm\b", r"\bweather\b", r"\bprecipitation\b",
    r"\bdrought\b", r"\bheatwave\b", r"\bfrost\b", r"\bwildfire\b",
]

_CRYPTO_KEYWORDS = [
    r"\bbitcoin\b", r"\bbtc\b", r"\bethereum\b", r"\beth\b", r"\bcrypto\b",
    r"\bblockchain\b", r"\bdefi\b", r"\bnft\b", r"\bsolana\b", r"\bcoinbase\b",
]

_DOMAIN_PATTERNS: List[Tuple[str, List[str]]] = [
    ("sports",    _SPORTS_KEYWORDS),
    ("economics", _ECONOMICS_KEYWORDS),
    ("politics",  _POLITICS_KEYWORDS),
    ("weather",   _WEATHER_KEYWORDS),
    ("crypto",    _CRYPTO_KEYWORDS),
]

# ── Tool domain map ────────────────────────────────────────────────────────────

DOMAIN_TOOLS: dict[str, List[str]] = {
    "sports": [
        "sportsbook_implied_probability_tool",
        "snapshot_volatility_tool",
        "spread_compression_tool",
        "price_jump_detector_tool",
        "liquidity_spike_tool",
    ],
    "economics": [
        "fred_macro_tool",
        "bls_labor_tool",
        "snapshot_volatility_tool",
        "spread_compression_tool",
        "price_jump_detector_tool",
    ],
    "politics": [
        "fred_macro_tool",
        "snapshot_volatility_tool",
        "price_jump_detector_tool",
        "liquidity_spike_tool",
    ],
    "weather": [
        "weather_probability_tool",
        "snapshot_volatility_tool",
    ],
    "crypto": [
        "snapshot_volatility_tool",
        "spread_compression_tool",
        "price_jump_detector_tool",
        "liquidity_spike_tool",
    ],
    "other": [
        "snapshot_volatility_tool",
        "spread_compression_tool",
        "price_jump_detector_tool",
    ],
}

# For outdoor venues, add weather tool to sports
_OUTDOOR_SPORTS_KEYWORDS = [
    r"\bfootball\b", r"\bnfl\b", r"\bbaseball\b", r"\bmlb\b",
    r"\bsoccer\b", r"\bgolf\b", r"\btennis\b", r"\boutdoor\b",
]


def classify_market(market_id: str, market_title: str) -> str:
    """
    Classify a Kalshi market into a domain.

    Args:
        market_id:    e.g. "STUB-NBA-LAL-BOS-001"
        market_title: e.g. "Lakers vs Celtics: Lakers Win"

    Returns:
        Domain string: "sports" | "economics" | "politics" | "weather" | "crypto" | "other"
    """
    text = f"{market_id} {market_title}".lower()

    for domain, patterns in _DOMAIN_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return domain

    return "other"


def get_domain_tools(domain: str, available_tools: List[str]) -> List[str]:
    """
    Return the subset of available_tools relevant for the given domain.

    Falls back to all available_tools if domain is unrecognised.

    Args:
        domain:          Output of classify_market().
        available_tools: Names of all tools currently in the registry.

    Returns:
        Filtered list of tool names appropriate for this domain.
    """
    domain_allowlist = DOMAIN_TOOLS.get(domain, DOMAIN_TOOLS["other"])
    filtered = [t for t in available_tools if t in domain_allowlist]

    # Safety: always include at least one tool
    if not filtered:
        filtered = available_tools[:3] if len(available_tools) >= 3 else available_tools

    return filtered


def is_outdoor_sport(market_id: str, market_title: str) -> bool:
    """Return True if the market looks like an outdoor sport (add weather tool)."""
    text = f"{market_id} {market_title}".lower()
    return any(re.search(p, text) for p in _OUTDOOR_SPORTS_KEYWORDS)


def get_domain_tools_with_weather(domain: str, available_tools: List[str],
                                   market_id: str, market_title: str) -> List[str]:
    """Like get_domain_tools but adds weather_probability_tool for outdoor sports."""
    tools = get_domain_tools(domain, available_tools)
    if domain == "sports" and is_outdoor_sport(market_id, market_title):
        if "weather_probability_tool" not in tools and "weather_probability_tool" in available_tools:
            tools.append("weather_probability_tool")
    return tools
