-- Structured memory: events (time-series) + lists (collections).
-- Complements the file-based MD memory (durable per-person facts) and
-- the memory_index vector cache (semantic search over everything).

-- Events: timestamped occurrences. Workouts, meals, moods, observations,
-- anything that has a "when" and accumulates over time.
CREATE TABLE IF NOT EXISTS memory_events (
    id           SERIAL PRIMARY KEY,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    category     TEXT NOT NULL,           -- 'workout', 'meal', 'mood', 'observation', ...
    person       TEXT,                    -- who it's about (NULL = household-level)
    content      TEXT NOT NULL,           -- short natural-language description
    metadata     JSONB,                   -- structured details (sets/reps, ingredients, etc.)
    source       TEXT,                    -- 'signal', 'voice', 'manual', etc.
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS memory_events_person_time
    ON memory_events (person, occurred_at DESC);
CREATE INDEX IF NOT EXISTS memory_events_category_time
    ON memory_events (category, occurred_at DESC);


-- Lists: named collections. Mother's Day ideas, gift ideas, books to read,
-- household projects, packing lists for trips, etc.
CREATE TABLE IF NOT EXISTS memory_lists (
    id           SERIAL PRIMARY KEY,
    name         TEXT UNIQUE NOT NULL,    -- slugged name: 'mothers_day_2026', 'cole_birthday_gifts'
    title        TEXT,                    -- human title: "Mother's Day 2026 ideas"
    description  TEXT,
    created_by   TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    archived_at  TIMESTAMPTZ
);


CREATE TABLE IF NOT EXISTS memory_list_items (
    id           SERIAL PRIMARY KEY,
    list_id      INTEGER NOT NULL REFERENCES memory_lists(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,
    metadata     JSONB,
    added_by     TEXT,
    added_at     TIMESTAMPTZ DEFAULT NOW(),
    done         BOOLEAN NOT NULL DEFAULT FALSE,
    done_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS memory_list_items_list
    ON memory_list_items (list_id, added_at DESC);
