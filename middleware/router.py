"""Decide which inference tier handles a request.

- Tier 1 (`local`): fast local Llama 3.1 70B for routine asks.
- Tier 2 (`claude`): Claude Opus 4.7 via CLI for research, code, vision.
- Tier 2-xhigh (`claude_xhigh`): same model, max effort, for hardest tasks
  (multi-constraint meal planning, deep household analysis, etc.).

The router uses keyword + length heuristics. It deliberately starts simple;
we'll tune from real conversation logs.
"""
from __future__ import annotations

import re

# Patterns that strongly suggest Claude Opus is the right tool.
_CLAUDE_PATTERNS = re.compile(
    r"\b("
    r"plan(?:s)?\s+(?:the|a|my|our)?\s*(?:week|month|year|days?\s+of)|"
    r"draft|write\s+(?:an?\s+)?(?:email|letter|memo|message|note|report)|"
    r"research|deep\s*think|analy[sz]e|compare\s+\w+\s+and\s+\w+|"
    r"strategy|optimi[sz]e|architect|design\s+(?:a|the|my|our)|"
    r"explain\s+(?:in\s+detail|deeply|thoroughly)|why\s+does"
    r")\b",
    re.IGNORECASE,
)

# Hardest-tier indicators — multi-constraint planning, broad coordination.
_XHIGH_PATTERNS = re.compile(
    r"\b("
    r"plan\s+(?:meals?|dinners?|breakfasts?|lunches?)\s+(?:for|across|using|considering)|"
    r"meal\s*plan|"
    r"week\s+of\s+(?:dinners?|meals?|menus?)|"
    r"refinanc(?:e|ing)|"
    r"coordinate\s+(?:across|between|among)|"
    r"taking\s+into\s+account|"
    r"considering\s+(?:our|the|my)\s+(?:budget|schedule|preferences|constraints)"
    r")\b",
    re.IGNORECASE,
)


def classify_request(text: str) -> str:
    """Returns 'local' | 'claude' | 'claude_xhigh'."""
    if not text:
        return "local"
    if _XHIGH_PATTERNS.search(text):
        return "claude_xhigh"
    if _CLAUDE_PATTERNS.search(text):
        return "claude"
    # Length-based escalation: very long asks usually want Claude.
    if len(text.split()) > 60:
        return "claude"
    return "local"
