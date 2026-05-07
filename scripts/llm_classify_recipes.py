"""Sonnet pass over recipes that the heuristic enricher couldn't
classify — fills in course / dish_type / tags from title + first
few ingredients.

Idempotent: only updates rows where the classification produces a
non-null value AND the existing value is null. Won't overwrite
anything Casey curated.
"""
from __future__ import annotations
import asyncio, json, os, re, sys, time, subprocess

sys.path.insert(0, "/opt/benson/middleware")
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from config import PG_DSN

CLAUDE_BIN = "/opt/benson/middleware/venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude"


def claude_print(prompt: str, model: str = "sonnet", timeout_s: int = 240) -> str:
    """Call the bundled CLI in --print mode (NOT the SDK's stream-JSON
    pipeline). The SDK path keeps timing out on 'Control request
    timeout: initialize' for batch scripts; --print mode is reliable
    when invoked from a writable cwd with HOME set to /home/casey."""
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # force OAuth path
    env["HOME"] = "/home/casey"
    out = subprocess.run(
        [CLAUDE_BIN, "-p", prompt, "--model", model,
         "--permission-mode", "bypassPermissions"],
        cwd="/tmp",
        env=env,
        capture_output=True, text=True, timeout=timeout_s,
    )
    if out.returncode != 0:
        raise RuntimeError(f"claude exit {out.returncode}: stderr={out.stderr[:300]}")
    return out.stdout


PROMPT = """You classify household recipes into a fixed set of categories.

For each recipe in the INPUT list, return a JSON object with the same `id`
plus three fields:
  "course":     one of "Main", "Side", "Dessert", "Sauce", "Other" (use Other only if nothing fits)
  "dish_type":  one short snake_case label like "pasta", "soup", "salad",
                "casserole", "stir_fry", "stew", "sandwich", "pizza", "taco",
                "bread", "breakfast", "dessert", "drink", "snack",
                "roasted_meat", "grilled_meat", "seafood",
                "vegetarian_main", "side_veg", "side_starch", "sauce", "dip",
                "marinade", "dressing". Pick the closest single tag.
  "tags":       a list of 0-4 short lowercase tags from this set:
                cuisines: italian, mexican, asian, indian, chinese, japanese,
                  thai, vietnamese, french, mediterranean, middle_eastern,
                  greek, american, southern, cajun, caribbean, latin, german,
                  british
                diet: vegetarian, vegan, gluten_free, dairy_free, low_carb,
                  high_protein
                method: slow_cooker, instant_pot, sheet_pan, one_pot,
                  no_bake, grilled, fried, air_fryer
                Don't invent tags outside this set.

Output: a SINGLE JSON array, no markdown fences, no commentary.

INPUT:
INPUT_JSON_PLACEHOLDER
"""


def fetch_unclassified() -> list[dict]:
    with psycopg2.connect(**PG_DSN) as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, title, course, dish_type, tags, ingredients
            FROM recipes
            WHERE course IS NULL OR dish_type IS NULL
            ORDER BY id
            """
        )
        return [dict(r) for r in cur.fetchall()]


def short_ingredients(ing) -> str:
    if not isinstance(ing, list):
        return ""
    parts = []
    for item in ing[:6]:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(item.get("text") or item.get("name") or "")
    return " | ".join(p for p in parts if p)[:300]


async def classify_batch(items: list[dict]) -> list[dict]:
    payload = [
        {"id": r["id"], "title": r["title"], "ingredients": short_ingredients(r.get("ingredients"))}
        for r in items
    ]
    prompt = PROMPT.replace("INPUT_JSON_PLACEHOLDER", json.dumps(payload, indent=1))
    try:
        raw = await asyncio.to_thread(claude_print, prompt, "sonnet", 240)
    except Exception as e:
        print(f"  !! claude failed: {e}")
        return []
    if not raw:
        return []
    blob = raw.strip()
    if blob.startswith("```"):
        blob = blob.split("\n", 1)[1] if "\n" in blob else blob
        if blob.endswith("```"):
            blob = blob.rsplit("```", 1)[0]
    try:
        return json.loads(blob.strip())
    except json.JSONDecodeError as e:
        print(f"  !! parse error: {e}; first 300 chars: {blob[:300]!r}")
        return []


def apply_classification(items: list[dict], rows_by_id: dict[int, dict]) -> int:
    updated = 0
    with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
        for it in items:
            rid = it.get("id")
            if not rid or rid not in rows_by_id:
                continue
            current = rows_by_id[rid]
            new_course = it.get("course") if not current.get("course") else None
            new_dt = it.get("dish_type") if not current.get("dish_type") else None
            new_tags_in = it.get("tags") or []
            existing_tags = current.get("tags") or []
            if not isinstance(existing_tags, list):
                existing_tags = []
            merged_tags = sorted(set(existing_tags) | set(new_tags_in))
            tags_changed = merged_tags != sorted(existing_tags)

            if not (new_course or new_dt or tags_changed):
                continue

            sets = []
            params: list = []
            if new_course:
                sets.append("course=%s")
                params.append(new_course)
            if new_dt:
                sets.append("dish_type=%s")
                params.append(new_dt)
            if tags_changed:
                sets.append("tags=%s::jsonb")
                params.append(Json(merged_tags))
            params.append(rid)
            cur.execute(
                f"UPDATE recipes SET {', '.join(sets)} WHERE id=%s",
                tuple(params),
            )
            updated += 1
        c.commit()
    return updated


async def main():
    rows = fetch_unclassified()
    rows_by_id = {r["id"]: r for r in rows}
    print(f"unclassified: {len(rows)}")

    BATCH = 25
    total_updated = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i : i + BATCH]
        print(f"--- batch {i // BATCH + 1}: {len(batch)} rows")
        t0 = time.time()
        result = await classify_batch(batch)
        print(f"    Sonnet returned {len(result)} classifications in {time.time()-t0:.1f}s")
        if result:
            n = apply_classification(result, rows_by_id)
            total_updated += n
            print(f"    {n} updated")
    print()
    print(f"total updated: {total_updated}")
    with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM recipes WHERE course IS NULL OR dish_type IS NULL")
        rem = cur.fetchone()[0]
        print(f"still unclassified: {rem}")


if __name__ == "__main__":
    asyncio.run(main())
