"""Bulk-import recipes from sitemap-driven sources.

Strategy:
  1. Pull each source's sitemap-of-recipes
  2. Sample N URLs per source for diversity
  3. Scrape each via recipe-scrapers (handles 1000+ sites' JSON-LD)
  4. Dedupe against the existing recipes table (by URL exact + title fuzzy)
  5. Insert with title / source_url / ingredients / steps / image_url /
     prep_time / course / notes (description)

Run with --limit N to cap; --dry to skip DB writes.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

sys.path.insert(0, "/opt/benson/middleware")
import psycopg2
from psycopg2.extras import Json
from config import PG_DSN

USER_AGENT = "Mozilla/5.0 (compatible; BensonHomeHub/1.0; recipe importer)"
REQUEST_TIMEOUT = 20

SOURCES: list[tuple[str, str, str]] = [
    ("allrecipes",
     "https://www.allrecipes.com/sitemap_1.xml",
     r"^https://www\.allrecipes\.com/recipe/\d+/"),
    ("simplyrecipes",
     "https://www.simplyrecipes.com/sitemap.xml",
     r"^https://www\.simplyrecipes\.com/recipes/[^/]+/?$"),
    ("budgetbytes",
     "https://www.budgetbytes.com/sitemap_index.xml",
     r"^https://www\.budgetbytes\.com/[^/]+/?$"),
    ("seriouseats",
     "https://www.seriouseats.com/sitemap_1.xml",
     r"^https://www\.seriouseats\.com/[^/]+-recipe-\d+"),
    ("halfbakedharvest",
     "https://www.halfbakedharvest.com/sitemap_index.xml",
     r"^https://www\.halfbakedharvest\.com/[^/]+/?$"),
]


def _http_get(url: str, retries: int = 2) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise last


def fetch_sitemap(url: str) -> list[str]:
    body = _http_get(url)
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs: list[str] = []
    if root.tag.endswith("sitemapindex"):
        for s in root.findall("sm:sitemap/sm:loc", ns):
            try:
                locs.extend(fetch_sitemap(s.text.strip()))
            except Exception:
                continue
    else:
        for u in root.findall("sm:url/sm:loc", ns):
            if u.text:
                locs.append(u.text.strip())
    return locs


def existing_recipe_keys() -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    titles: set[str] = set()
    with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
        cur.execute("SELECT title, source_url FROM recipes")
        for title, url in cur.fetchall():
            if url:
                urls.add(url.split("?")[0].rstrip("/"))
            if title:
                titles.add(_normalize_title(title))
    return urls, titles


_TITLE_NOISE = re.compile(r"\b(recipe|the|a|easy|best|homemade|how to make)\b", re.I)


def _normalize_title(t: str) -> str:
    t = t.lower()
    t = _TITLE_NOISE.sub("", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def scrape_one(url: str) -> dict | None:
    try:
        from recipe_scrapers import scrape_html
        html = _http_get(url)
        scraper = scrape_html(html, org_url=url)
        title = (scraper.title() or "").strip()
        if not title:
            return None
        ingredients = scraper.ingredients() or []
        steps_raw = scraper.instructions() or ""
        if isinstance(steps_raw, str):
            steps = [s.strip() for s in steps_raw.split("\n") if s.strip()]
        else:
            steps = list(steps_raw)
        image = None
        try:
            image = scraper.image()
        except Exception:
            pass
        prep_time = None
        for fn in ("total_time", "prep_time", "cook_time"):
            try:
                v = getattr(scraper, fn, lambda: None)()
                if v:
                    prep_time = int(v)
                    break
            except Exception:
                continue
        description = ""
        try:
            description = (scraper.description() or "")[:1000]
        except Exception:
            pass
        return {
            "title": title,
            "source_url": url,
            "ingredients": [{"text": s} for s in ingredients],
            "steps": steps,
            "image_url": image,
            "prep_time": prep_time,
            "notes": description,
        }
    except Exception:
        return None


def insert_recipe(rec: dict) -> int | None:
    with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recipes
                (title, source, source_url, ingredients, steps, tags,
                 image_url, prep_time, notes)
            VALUES (%s, 'web_import', %s, %s, %s, %s::jsonb, %s, %s, %s)
            RETURNING id
            """,
            (
                rec["title"],
                rec["source_url"],
                Json(rec["ingredients"]),
                Json(rec["steps"]),
                Json([]),
                rec.get("image_url"),
                rec.get("prep_time"),
                rec.get("notes") or None,
            ),
        )
        return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--per-site", type=int, default=100)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--rate", type=float, default=2.0)
    args = ap.parse_args()

    print(f"=== bulk recipe import — target: {args.limit}, dry={args.dry} ===")
    seen_urls, seen_titles = existing_recipe_keys()
    print(f"existing: {len(seen_urls)} URLs, {len(seen_titles)} normalized titles")

    candidates: list[tuple[str, str]] = []
    for label, sitemap_url, pattern in SOURCES:
        try:
            urls = fetch_sitemap(sitemap_url)
        except Exception as e:
            print(f"  [{label}] sitemap fetch failed: {e}")
            continue
        rx = re.compile(pattern)
        urls = [u for u in urls if rx.match(u)]
        random.shuffle(urls)
        urls = urls[: args.per_site * 2]
        print(f"  [{label}] {len(urls)} candidate URLs")
        candidates.extend((label, u) for u in urls)

    random.shuffle(candidates)
    print(f"total candidates: {len(candidates)}")

    inserted = 0
    skipped_dup = 0
    failed = 0
    by_source: dict[str, int] = {}
    site_caps: dict[str, int] = {}

    for label, url in candidates:
        if inserted >= args.limit:
            break
        if site_caps.get(label, 0) >= args.per_site:
            continue
        canonical = url.split("?")[0].rstrip("/")
        if canonical in seen_urls:
            skipped_dup += 1
            continue

        rec = scrape_one(url)
        if not rec:
            failed += 1
            time.sleep(args.rate)
            continue
        norm = _normalize_title(rec["title"])
        if norm in seen_titles:
            skipped_dup += 1
            time.sleep(args.rate)
            continue
        seen_titles.add(norm)
        seen_urls.add(canonical)

        if not args.dry:
            try:
                rid = insert_recipe(rec)
            except Exception as e:
                print(f"  [{label}] insert failed: {e}")
                failed += 1
                time.sleep(args.rate)
                continue
            print(f"  +{rid:4d}  [{label}] {rec['title'][:70]}")
        else:
            print(f"  (dry) [{label}] {rec['title'][:70]}")
        inserted += 1
        by_source[label] = by_source.get(label, 0) + 1
        site_caps[label] = site_caps.get(label, 0) + 1
        time.sleep(args.rate)

        if inserted and inserted % 25 == 0:
            print(f"--- {inserted} imported · {failed} failed · {skipped_dup} dup ---")

    print()
    print(f"=== done: {inserted} imported, {failed} failed, {skipped_dup} dup ===")
    for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()
