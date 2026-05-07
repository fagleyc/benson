"""One-off heuristic enrichment of the recipes table.

Fills in dish_type, tags, and missing course based on title +
source_url + ingredient strings. Idempotent — re-running only
overwrites rows that don't already have a populated value.

Reversible: the original values are preserved unless empty.
"""
import sys, json, re
sys.path.insert(0, "/opt/benson/middleware")
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from config import PG_DSN


# Title-keyword → (dish_type, course_hint, cuisine_tag)
# Order matters — first match wins, so put more-specific ahead of generic.
RULES = [
    # Specific dish_type patterns (most specific first)
    (r"\bpizza\b",              "pizza",      "Main",     "italian"),
    (r"\bbolognese\b|\blasagn", "pasta",      "Main",     "italian"),
    (r"\b(spaghetti|fettuccin|linguin|penne|rigaton|orzo|risotto|carbonara|alfredo|gnocchi|ravioli)\b",
                                "pasta",      "Main",     "italian"),
    (r"\bpasta\b",              "pasta",      "Main",     "italian"),
    (r"\b(taco|burrito|enchilada|quesadilla|fajita|carnitas|tortilla|chimichanga)\b",
                                "mexican",    "Main",     "mexican"),
    (r"\b(curry|tikka|masala|biryani|naan|samosa|dal|paneer|vindaloo|saag)\b",
                                "indian",     "Main",     "indian"),
    (r"\b(stir.?fry|teriyaki|tempura|sushi|ramen|udon|pho|pad thai|kung pao|lo mein|bulgogi|bibimbap|gyoza|dumpling)\b",
                                "asian",      "Main",     "asian"),
    (r"\b(soup|chowder|chili|stew|bisque|broth)\b",
                                "soup",       "Main",     None),
    (r"\b(salad|slaw|coleslaw)\b",
                                "salad",      "Side",     None),
    (r"\b(sandwich|burger|wrap|panini|sub|hoagie|grinder|melt)\b",
                                "sandwich",   "Main",     None),
    (r"\b(casserole|bake|gratin)\b",
                                "casserole",  "Main",     None),
    (r"\b(grill|grilled|bbq|barbecue)\b",
                                "grilled",    "Main",     None),
    (r"\b(roast|roasted)\b",
                                "roasted",    "Main",     None),
    (r"\b(pancake|waffle|french toast|omelette|omelet|frittata|quiche|breakfast|granola|oatmeal|hash brown|scrambled)\b",
                                "breakfast",  "Other",    None),
    (r"\b(cake|pie|cookie|brownie|cupcake|muffin|tart|crumble|cobbler|pudding|mousse|ice cream|sorbet|cheesecake)\b",
                                "dessert",    "Dessert",  None),
    (r"\b(bread|loaf|biscuit|scone|focaccia|rolls?|baguette|sourdough|cornbread)\b",
                                "bread",      "Side",     None),
    (r"\b(sauce|dressing|marinade|aioli|pesto|salsa|chutney|relish)\b",
                                "sauce",      "Sauce",    None),
    (r"\b(rice|quinoa|couscous|polenta|grits)\b",
                                "grain",      "Side",     None),
    # Generic protein hints (only set dish_type, no course override)
    (r"\b(chicken|poultry)\b",  "chicken",    None,       None),
    (r"\b(beef|steak|brisket|short rib)\b",
                                "beef",       None,       None),
    (r"\b(pork|bacon|ham|sausage|chop)\b",
                                "pork",       None,       None),
    (r"\b(salmon|tuna|cod|tilapia|shrimp|scallop|crab|lobster|fish|seafood)\b",
                                "seafood",    None,       None),
    (r"\b(tofu|tempeh|seitan|lentil|chickpea|veggie|vegetarian|vegan)\b",
                                "vegetarian", None,       "vegetarian"),
]

# Source-url cuisine hints (override/add when title is ambiguous)
SOURCE_HINTS = [
    (r"simplyrecipes|allrecipes|food52|seriouseats", None),
    (r"natashaskitchen|olgasflavorfactory", "russian"),
    (r"halfbakedharvest", None),
    (r"cooking\.nytimes|nytimes", None),
    (r"foodnetwork", None),
    (r"thekitchn",                None),
    (r"epicurious",               None),
    (r"bonappetit|bonappétit",    None),
    (r"americastestkitchen|cookscountry|cooksillustrated", None),
    (r"tasty|buzzfeed",           None),
    (r"gimmesomeoven",            None),
    (r"smittenkitchen",           None),
    (r"budgetbytes",              None),
]


def classify(title: str, source_url: str | None, ingredients_text: str | None) -> dict:
    """Return {dish_type, tags, course_hint} for one recipe."""
    haystack = " ".join([
        (title or "").lower(),
        (source_url or "").lower(),
        (ingredients_text or "").lower()[:600],  # cap to avoid long ingredient lists swamping
    ])

    dish_type = None
    course_hint = None
    tags: set[str] = set()

    for pattern, dt, ch, cuisine in RULES:
        if re.search(pattern, haystack):
            if dish_type is None:
                dish_type = dt
            if course_hint is None and ch:
                course_hint = ch
            if cuisine:
                tags.add(cuisine)

    return {
        "dish_type": dish_type,
        "course_hint": course_hint,
        "tags": sorted(tags),
    }


def main():
    updated = 0
    skipped = 0
    with psycopg2.connect(**PG_DSN) as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, title, course, dish_type, tags, source_url, ingredients FROM recipes"
        )
        rows = cur.fetchall()

    for r in rows:
        # Build a short ingredient text blob from the JSONB list.
        ing = r.get("ingredients") or []
        ing_text = ""
        if isinstance(ing, list):
            parts = []
            for item in ing[:15]:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text") or item.get("name") or "")
            ing_text = " ".join(parts)

        result = classify(r["title"], r.get("source_url"), ing_text)

        # Only fill in missing values; don't overwrite anything Casey
        # has already curated.
        new_dish_type = r.get("dish_type") or result["dish_type"]
        existing_course = (r.get("course") or "").strip()
        new_course = existing_course or result["course_hint"] or None

        existing_tags = r.get("tags") or []
        if not isinstance(existing_tags, list):
            existing_tags = []
        merged_tags = sorted(set(existing_tags) | set(result["tags"]))

        # Bail out cheaply if nothing changed.
        if (new_dish_type == r.get("dish_type")
                and new_course == r.get("course")
                and merged_tags == sorted(existing_tags)):
            skipped += 1
            continue

        with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
            cur.execute(
                "UPDATE recipes SET dish_type=%s, course=%s, tags=%s::jsonb WHERE id=%s",
                (new_dish_type, new_course, Json(merged_tags), r["id"]),
            )
        updated += 1

    print(f"updated: {updated}")
    print(f"skipped (no change): {skipped}")
    print()
    print("=== course distribution after enrichment ===")
    with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(course,'(none)') AS course, COUNT(*) "
            "FROM recipes GROUP BY course ORDER BY COUNT(*) DESC"
        )
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]}")
        cur.execute(
            "SELECT COALESCE(dish_type,'(none)') AS dish_type, COUNT(*) "
            "FROM recipes GROUP BY dish_type ORDER BY COUNT(*) DESC"
        )
        print()
        print("=== dish_type distribution ===")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]}")


if __name__ == "__main__":
    main()
