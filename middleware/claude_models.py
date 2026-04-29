"""Model router for Benson's Claude tier.

A single decision point: given user text + (optional) intent type, return
which Claude model + thinking budget to use.

Tiers:
  - HAIKU   (claude-haiku-4-5)   : ~700ms-2s. Default chat, HA confirmations,
                                    short announcements, memory extraction.
  - SONNET  (claude-sonnet-4-6)  : ~3-6s. Recipe vision, recipe transcript
                                    extraction, structured output, code,
                                    drafts. Vision-capable.
  - OPUS    (claude-opus-4-7)    : ~10-30s with thinking. Multi-constraint
                                    planning, deep research, architecture.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ModelTier(str, Enum):
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


# API model IDs.
MODEL_ID = {
    ModelTier.HAIKU: "claude-haiku-4-5-20251001",
    ModelTier.SONNET: "claude-sonnet-4-6",
    ModelTier.OPUS: "claude-opus-4-7",
}


# Thinking-token budgets per tier (Sonnet+Opus support extended thinking).
# Haiku doesn't use thinking. Opus xhigh gets a large budget.
THINKING_BUDGET = {
    ModelTier.HAIKU: 0,
    ModelTier.SONNET: 0,        # set per-call when needed
    ModelTier.OPUS: 16000,      # default for OPUS in this app
}


@dataclass
class ModelChoice:
    tier: ModelTier
    model_id: str
    thinking_tokens: int = 0
    max_tokens: int = 1024
    rationale: str = ""

    @property
    def label(self) -> str:
        return self.tier.value


# ─── Pattern catalogues ──────────────────────────────────────────────────
# OPUS triggers — multi-constraint, multi-step, high-stakes deliberation.
_DET = r"(?:the\s+|a\s+|an\s+|my\s+|our\s+|this\s+|next\s+)?"
_OPUS_PATTERNS = re.compile(
    r"\b("
    rf"plan\s+{_DET}(?:meals?|dinners?|breakfasts?|lunches?|menus?|week)|"
    r"meal\s*plan|"
    r"week\s+of\s+(?:dinners?|meals?|menus?)|"
    r"refinanc(?:e|ing)|"
    r"coordinate\s+(?:across|between|among)|"
    r"taking\s+into\s+account|"
    r"considering\s+(?:our|the|my)\s+(?:budget|schedule|preferences|constraints)|"
    r"research\s+the\s+best|"
    r"deep\s+(?:think|analysis|dive)|"
    r"long[\-\s]term\s+(?:strategy|plan)|"
    r"trade[\-\s]offs?\s+between|"
    r"design\s+(?:the|a|an?|our|my)\s+(?:architecture|system|approach|strategy)"
    r")\b",
    re.IGNORECASE,
)

# SONNET triggers — single-step but needs reasoning, structure, drafts, code.
_SONNET_PATTERNS = re.compile(
    r"\b("
    # Drafting / writing
    r"draft\s+(?:an?\s+)?(?:email|letter|memo|message|note|report)|"
    r"write\s+(?:python|code|a\s+function|a\s+script|an?\s+email|a\s+story|the\s+code)|"
    # Explanation / analysis
    r"explain\s+in\s+detail|explain\s+thoroughly|deep\s+dive|"
    r"compare\s+\w+\s+and\s+\w+|"
    r"summari[sz]e\s+(?:this|the\s+following|the\s+article|the\s+paper)|"
    r"analy[sz]e|"
    r"review\s+(?:my|this|the)\s+(?:code|writing|paper|draft)|"
    # Diagnostic / why-did / what-went-wrong — these are where Haiku
    # produced confidently-wrong answers in production. Routing them to
    # Sonnet so the investigation actually grounds in source/logs.
    r"why\s+(?:did|does|is|isn'?t|can'?t|won'?t)|"
    r"what\s+(?:went\s+wrong|happened|broke|failed|caused)|"
    r"diagnose|debug|troubleshoot|investigate|"
    r"how\s+come|how\s+do(?:es)?\s+(?:this|that|it)\s+work|"
    # Self-modification / fix this / propose a change
    r"fix\s+(?:the|this|that|your|my|the\s+\w+)|"
    r"broken|isn'?t\s+working|not\s+working|stopped\s+working|doesn'?t\s+work|"
    r"propose\s+(?:a\s+)?change|open\s+a\s+proposal|"
    r"refactor|rewrite|improve\s+(?:the|your)\s+(?:code|implementation)|"
    # Multi-tool orchestration cues — when the user wants Benson to
    # compose tools (vision + signal, grep + propose, etc.) Haiku tends
    # to skip steps. Match 'send' and an attachment-like noun nearby.
    r"send\s+(?:\w+\s+){0,4}(?:image|photo|file|attachment|screenshot|picture|snapshot)|"
    r"attach\s+(?:an?|the|my)\s+(?:image|photo|file|screenshot|picture)|"
    r"(?:read|grep|search)\s+(?:your|the|my)\s+(?:source|code|logs|history)"
    r")\b",
    re.IGNORECASE,
)


def select(
    user_text: str,
    *,
    intent_type: str | None = None,
    is_compose_announce: bool = False,
) -> ModelChoice:
    """Pick a model based on user text and intent context."""
    if intent_type == "vision":
        return ModelChoice(
            tier=ModelTier.SONNET,
            model_id=MODEL_ID[ModelTier.SONNET],
            max_tokens=4096,
            rationale="vision needed; Sonnet is vision-capable and sufficient",
        )

    if intent_type == "memory_extraction":
        return ModelChoice(
            tier=ModelTier.HAIKU,
            model_id=MODEL_ID[ModelTier.HAIKU],
            max_tokens=512,
            rationale="lightweight extraction task; Haiku handles it cheaply",
        )

    if is_compose_announce:
        return ModelChoice(
            tier=ModelTier.HAIKU,
            model_id=MODEL_ID[ModelTier.HAIKU],
            max_tokens=400,
            rationale="3-sentence Sonos announcement; Haiku is plenty",
        )

    text = user_text or ""

    if _OPUS_PATTERNS.search(text):
        return ModelChoice(
            tier=ModelTier.OPUS,
            model_id=MODEL_ID[ModelTier.OPUS],
            thinking_tokens=THINKING_BUDGET[ModelTier.OPUS],
            max_tokens=4096,
            rationale="multi-constraint planning / deep deliberation",
        )

    if _SONNET_PATTERNS.search(text) or len(text.split()) > 60:
        return ModelChoice(
            tier=ModelTier.SONNET,
            model_id=MODEL_ID[ModelTier.SONNET],
            max_tokens=2048,
            rationale="needs structured output, draft, or code",
        )

    return ModelChoice(
        tier=ModelTier.HAIKU,
        model_id=MODEL_ID[ModelTier.HAIKU],
        max_tokens=800,
        rationale="default chat / household Q&A",
    )
