-- /opt/benson/scripts/phase1_schema.sql
-- Phase 1.2 — Benson database schema.
-- Run as the `benson` user against the `benson` database after the
-- vector extension has been created by a superuser.

-- ─── Memories (semantic memory layer) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS memories (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(1024),                 -- bge-large-en-v1.5
    source      TEXT,                          -- voice, telegram, auto, seed
    speaker     TEXT,                          -- household member or null
    room        TEXT,                          -- which satellite heard it
    importance  REAL DEFAULT 0.5,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memories_source_idx ON memories (source);
CREATE INDEX IF NOT EXISTS memories_created_at_idx ON memories (created_at DESC);

-- ─── Recipes (Benson + carried forward from miniserver:/opt/recipeapp) ─
CREATE TABLE IF NOT EXISTS recipes (
    id                SERIAL PRIMARY KEY,
    title             TEXT NOT NULL,
    source            TEXT,                    -- photo, tiktok, web, manual, migrated
    source_url        TEXT,
    ingredients       JSONB,                   -- list of {text, name?, quantity?, unit?}
    steps             JSONB,                   -- list of strings
    tags              JSONB,                   -- list of strings
    image_path        TEXT,
    image_url         TEXT,                    -- original web URL (preserved from existing)
    household_rating  INTEGER,                 -- aggregate, 1-5
    user_rating       REAL,                    -- preserved from existing
    user_comments     TEXT,                    -- preserved from existing
    dish_type         TEXT,                    -- preserved from existing
    course            TEXT,                    -- preserved from existing
    prep_time         INTEGER,                 -- minutes
    parse_status      TEXT DEFAULT 'approved', -- preserved from existing
    last_made         DATE,
    notes             TEXT,
    legacy_recipe_id  INTEGER UNIQUE,          -- maps back to old recipe_id during migration
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS recipes_course_idx ON recipes (course);
CREATE INDEX IF NOT EXISTS recipes_source_idx ON recipes (source);

-- ─── Weekly meal plan (carried forward) ───────────────────────────────
CREATE TABLE IF NOT EXISTS weekly_plan (
    plan_date  DATE PRIMARY KEY,
    recipe_id  INTEGER REFERENCES recipes(id) ON DELETE SET NULL,
    status     TEXT
);

-- ─── Chores (carried forward — Cole and Zander, actively used) ────────
CREATE TABLE IF NOT EXISTS chores (
    id          SERIAL PRIMARY KEY,
    person      TEXT NOT NULL,
    chore_date  DATE,
    chore_name  TEXT,
    done        BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS chores_person_date_idx ON chores (person, chore_date);

-- ─── Conversations (Phase 4 logs) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    speaker         TEXT,
    room            TEXT,
    user_text       TEXT,
    benson_response TEXT,
    tier            TEXT,                       -- local, claude, claude_xhigh
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS conversations_created_at_idx ON conversations (created_at DESC);

-- ─── Household profile (versioned snapshots) ──────────────────────────
CREATE TABLE IF NOT EXISTS household_profile (
    id          SERIAL PRIMARY KEY,
    profile     JSONB NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
