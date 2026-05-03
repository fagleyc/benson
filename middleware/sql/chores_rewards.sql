-- chores_rewards.sql — 2026-05-03
-- Cole earns dollars per chore, Zander earns points per chore. Per-row
-- values let the same chore have different rewards for different
-- assignments. chore_templates is the catalog of reusable assignments
-- with default rewards, seeded from chores_archive's history.

ALTER TABLE chores
    ADD COLUMN IF NOT EXISTS dollars NUMERIC(8,2) DEFAULT 0
    CHECK (dollars IS NULL OR dollars >= 0);
ALTER TABLE chores
    ADD COLUMN IF NOT EXISTS points INTEGER DEFAULT 0
    CHECK (points IS NULL OR points >= 0);

CREATE TABLE IF NOT EXISTS chore_templates (
    id SERIAL PRIMARY KEY,
    person TEXT NOT NULL,
    chore_name TEXT NOT NULL,
    default_dollars NUMERIC(8,2) DEFAULT 0,
    default_points INTEGER DEFAULT 0,
    category TEXT,
    use_count INTEGER DEFAULT 0,
    archived_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (person, chore_name)
);
CREATE INDEX IF NOT EXISTS chore_templates_person_idx ON chore_templates(person);

-- Seed from history (idempotent — re-runs upsert use_count + archived_at).
INSERT INTO chore_templates (person, chore_name, use_count, archived_at)
SELECT
    person,
    LOWER(TRIM(REGEXP_REPLACE(chore_name, '\s+', ' ', 'g'))) AS chore_name,
    COUNT(*) AS use_count,
    MAX(chore_date) AS archived_at
FROM chores_archive
WHERE chore_name IS NOT NULL AND TRIM(chore_name) != ''
GROUP BY person, LOWER(TRIM(REGEXP_REPLACE(chore_name, '\s+', ' ', 'g')))
ON CONFLICT (person, chore_name) DO UPDATE SET
    use_count = EXCLUDED.use_count,
    archived_at = EXCLUDED.archived_at;
