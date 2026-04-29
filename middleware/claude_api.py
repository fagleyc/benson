"""Async Claude SDK wrapper.

Single entry point: ask(prompt, system_prompt, *, choice, image=None).

Used for:
  - Recipe vision (Sonnet, photo extraction)
  - Memory auto-extraction (Haiku, ~110 tokens per chat)
  - Fallback when the OAuth agent path fails

On API failure, returns a graceful error tuple instead of falling back
further (Ollama removed 2026-04-26). Caller decides what to do.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

import anthropic
from anthropic import AsyncAnthropic

from claude_models import ModelChoice, ModelTier, select as select_model

logger = logging.getLogger("benson.claude")


@dataclass
class VisionImage:
    media_type: str  # "image/jpeg" / "image/png" / "image/webp" / "image/gif"
    base64_data: str


_client_singleton: Optional[AsyncAnthropic] = None


def _client() -> AsyncAnthropic:
    """Lazy AsyncAnthropic with the API key from env."""
    global _client_singleton
    if _client_singleton is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in env")
        _client_singleton = AsyncAnthropic(api_key=key)
    return _client_singleton


async def ask(
    prompt: str,
    system_prompt: str,
    *,
    choice: ModelChoice | None = None,
    image: VisionImage | None = None,
    extra_context: str = "",
    timeout_s: float = 120.0,
) -> tuple[str, str]:
    """Run an Anthropic chat call. Falls back to Ollama on any error.

    Returns (response_text, tier_used). tier_used is one of:
      'haiku', 'sonnet', 'opus', 'ollama_fallback'.
    """
    if choice is None:
        choice = select_model(prompt, intent_type=("vision" if image else None))

    full_system = system_prompt
    if extra_context:
        full_system = f"{system_prompt}\n\n{extra_context}"

    user_blocks: list[dict] = []
    if image is not None:
        user_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": image.base64_data,
            },
        })
    user_blocks.append({"type": "text", "text": prompt})

    kwargs: dict = {
        "model": choice.model_id,
        "max_tokens": choice.max_tokens,
        "system": full_system,
        "messages": [{"role": "user", "content": user_blocks}],
    }
    if choice.thinking_tokens > 0:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": choice.thinking_tokens,
        }
        # Thinking budget eats into max_tokens; bump if needed.
        if choice.max_tokens <= choice.thinking_tokens:
            kwargs["max_tokens"] = choice.thinking_tokens + 2048

    try:
        client = _client()
        response = await asyncio.wait_for(
            client.messages.create(**kwargs), timeout=timeout_s
        )
        # Extract text from the first text block (skip thinking blocks).
        text_parts = [b.text for b in response.content if b.type == "text"]
        text = "\n".join(text_parts).strip()
        logger.info(
            f"claude {choice.tier.value} ok ({choice.rationale}): "
            f"{response.usage.input_tokens}+{response.usage.output_tokens} tok"
        )
        return text, choice.tier.value
    except (
        anthropic.APIError,
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        asyncio.TimeoutError,
    ) as e:
        logger.warning(f"claude {choice.tier.value} failed ({type(e).__name__}: {e})")
        return (
            f"Claude API call failed ({type(e).__name__}). "
            f"Check ANTHROPIC_API_KEY balance and network.",
            "api_failed",
        )
    except Exception as e:
        logger.exception(f"unexpected error from Claude API ({type(e).__name__})")
        return (
            f"Unexpected error from Claude API ({type(e).__name__}: {e}).",
            "api_failed",
        )
