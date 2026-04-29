"""Instacart cart-link generation.

Behind a Connect Developer key the API can create "shopping list" links
that pre-populate a cart. Without a key we fall back to a search URL that
the user can copy/paste — Phase 0 noted approval can take ~1 week.
"""
from __future__ import annotations

import logging
import os
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger("benson.instacart")

INSTACART_KEY = os.environ.get("INSTACART_API_KEY", "")
INSTACART_API = "https://connect.instacart.com/idp/v1/products/products_link"


class InstacartClient:
    async def create_shopping_link(self, items: list[str]) -> str:
        if not INSTACART_KEY:
            # Fallback: deep-link search for the first item; user pastes the rest.
            joined = " ".join(items[:3])
            return (
                f"https://www.instacart.com/store/s?k={quote_plus(joined)} "
                f"(API key not set; full cart import requires INSTACART_API_KEY)"
            )

        payload = {
            "title": "Benson grocery list",
            "image_url": "",
            "link_type": "shopping_list",
            "expires_in": 7,
            "instructions": [],
            "line_items": [
                {"name": it, "quantity": 1, "unit": "each"} for it in items
            ],
        }
        headers = {
            "Authorization": f"Bearer {INSTACART_KEY}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(INSTACART_API, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json().get("products_link_url", "")
