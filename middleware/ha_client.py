"""Thin async HA REST client for service calls and state reads."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config import HA_BASE_URL, HA_TOKEN

logger = logging.getLogger("benson.ha")


class HAUnavailable(RuntimeError):
    pass


def _headers() -> dict:
    if not HA_TOKEN:
        raise HAUnavailable("HA_LONG_LIVED_TOKEN not set in /etc/benson/env")
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


async def call_service(
    domain: str, service: str, data: dict | None = None,
    *, timeout_s: int = 15, return_response: bool = False,
) -> list[dict] | dict:
    """Fire a HA service call. Returns state changes (default) or the
    full response body when `return_response=True` (for services like
    `music_assistant.search` / `music_assistant.get_library` that
    return their result rather than mutating state)."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        url = f"{HA_BASE_URL}/api/services/{domain}/{service}"
        if return_response:
            url += "?return_response"
        resp = await client.post(url, headers=_headers(), json=data or {})
        if resp.status_code >= 400:
            raise HAUnavailable(
                f"HA service {domain}.{service} failed: {resp.status_code} {resp.text[:200]}"
            )
        return resp.json()


async def get_state(entity_id: str, *, timeout_s: int = 5) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(
            f"{HA_BASE_URL}/api/states/{entity_id}", headers=_headers()
        )
        if resp.status_code == 404:
            raise HAUnavailable(f"entity {entity_id} not found")
        resp.raise_for_status()
        return resp.json()


async def list_entities(domains: tuple[str, ...] | None = None) -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{HA_BASE_URL}/api/states", headers=_headers())
        resp.raise_for_status()
        states = resp.json()
        if domains:
            states = [s for s in states if s["entity_id"].split(".")[0] in domains]
        return states
