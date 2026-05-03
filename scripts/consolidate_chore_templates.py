"""Heuristic chore-template dedupe — token-set + synonyms + stemming.

Reversible: chore_templates was seeded from chores_archive (still
intact), so worst case re-run sql/chores_rewards.sql.
"""
import sys, re
sys.path.insert(0, "/opt/benson/middleware")
import psycopg2
from psycopg2.extras import RealDictCursor
from config import PG_DSN

# Words that don't carry meaning in chore names — stripped before
# comparing token sets.
_STOP = {
    "and", "the", "to", "from", "of", "in", "on", "at", "for",
    "a", "an", "with", "your", "my", "our", "all", "every",
    "around", "up", "down", "out", "back", "into", "off",
}

# Hand-coded synonyms — same chore, different wording. Each item is a
# canonical form that maps from a list of variants.
_SYNONYMS = {
    "trash": {"trash", "garbage", "trashcan", "trashcans", "rubbish", "garbage/recycle", "garbage/recycles"},
    "recycle": {"recycle", "recycling", "recycles", "recyclables"},
    "bbq": {"bbq", "grill", "barbeque"},
    "bluey": {"bluey", "blue", "dog"},
    "hottub": {"hottub", "hot-tub"},  # 'hot tub' → after token-strip becomes just 'tub'+'hot'
    "trashbin": {"trashbin", "trashbins", "bin", "bins"},
    "weed": {"weed", "weeds", "weeding"},
    "leaf": {"leaf", "leaves"},
    "dish": {"dish", "dishes", "dishwasher"},  # not perfect but close
    "yard": {"yard", "lawn", "backyard", "front-yard"},
    "boxe": {"box", "boxes"},
    "garbage": {"garbage", "trash"},  # alias both ways
}


def _canon_token(tok: str) -> str:
    """Map a token to its synonym group if any."""
    for canon, variants in _SYNONYMS.items():
        if tok in variants:
            return canon
    # Naive plural strip.
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokens(name: str) -> frozenset[str]:
    """Tokenize, drop stopwords, canonicalize via synonyms + plural-strip."""
    raw = re.sub(r"[^\w\s]+", " ", name.lower()).split()
    out = set()
    for t in raw:
        if t in _STOP or len(t) < 2:
            continue
        out.add(_canon_token(t))
    return frozenset(out)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster(rows: list[dict], threshold: float = 0.6) -> list[list[dict]]:
    """Single-link cluster by Jaccard similarity over canonical token sets."""
    items = [(r, _tokens(r["chore_name"])) for r in rows]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = _jaccard(items[i][1], items[j][1])
            if sim >= threshold:
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for idx, (row, _) in enumerate(items):
        groups.setdefault(find(idx), []).append(row)
    return list(groups.values())


def apply_clusters(person: str, clusters: list[list[dict]]) -> dict:
    """For each cluster: keep the highest-use_count member as canonical,
    sum use_counts, take max defaults, delete the rest."""
    kept, deleted, total_in = 0, 0, 0
    with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
        for group in clusters:
            total_in += len(group)
            group.sort(key=lambda r: (-r["use_count"], r["chore_name"]))
            canonical = group[0]
            members = group
            total_use = sum(m["use_count"] for m in members)
            max_dollars = max(
                (float(m.get("default_dollars") or 0) for m in members), default=0
            )
            max_points = max(
                (int(m.get("default_points") or 0) for m in members), default=0
            )
            cur.execute(
                "UPDATE chore_templates SET use_count=%s, default_dollars=%s, "
                "default_points=%s WHERE id=%s",
                (total_use, max_dollars, max_points, canonical["id"]),
            )
            kept += 1
            for m in members[1:]:
                cur.execute(
                    "DELETE FROM chore_templates WHERE id=%s", (m["id"],)
                )
                deleted += 1
        c.commit()
    return {"person": person, "in": total_in, "kept": kept, "deleted": deleted}


def main():
    with psycopg2.connect(**PG_DSN) as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, person, chore_name, use_count, default_dollars, "
            "default_points FROM chore_templates ORDER BY person, chore_name"
        )
        rows = [dict(r) for r in cur.fetchall()]

    by_person: dict[str, list] = {}
    for r in rows:
        by_person.setdefault(r["person"], []).append(r)

    grand = []
    for person, prows in by_person.items():
        clusters = cluster(prows, threshold=0.6)
        result = apply_clusters(person, clusters)
        print(f"{person}: {result['in']} → {result['kept']} (-{result['deleted']})")
        # Show the merges.
        for group in clusters:
            if len(group) <= 1:
                continue
            group.sort(key=lambda r: (-r["use_count"], r["chore_name"]))
            canon = group[0]["chore_name"]
            others = [g["chore_name"] for g in group[1:]]
            print(f"  '{canon}' ← {others}")
        grand.append(result)

    print()
    with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
        cur.execute("SELECT person, COUNT(*) FROM chore_templates GROUP BY person")
        for p, n in cur.fetchall():
            print(f"final {p}: {n}")


if __name__ == "__main__":
    main()
