-- Tier 1 autonomous self-modifications: trivial fixes Benson applied
-- directly without going through propose_change review.
--
-- Eligibility (enforced in self_modify.autofix): ≤5 files, ≤20 added+removed
-- lines, no blocklisted paths, only comment/docstring/log-string/markdown
-- changes, no AST structural delta. Anything else still routes through
-- propose_change for Casey to review on /admin/proposals.

CREATE TABLE IF NOT EXISTS autonomous_changes (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    rationale       TEXT NOT NULL,
    paths           TEXT[] NOT NULL,
    commit_sha      TEXT NOT NULL,
    diff_added      INT NOT NULL,
    diff_removed    INT NOT NULL,
    actor           TEXT NOT NULL DEFAULT 'benson-tier1',
    reverted_at     TIMESTAMPTZ,
    reverted_by     TEXT,
    revert_commit   TEXT
);

CREATE INDEX IF NOT EXISTS idx_autonomous_changes_created_at
    ON autonomous_changes (created_at DESC);
