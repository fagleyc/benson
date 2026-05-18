-- Scheduled household actions: persistent, process-independent scheduler.
--
-- Background: 2026-05-18 06:20 — a 6:20 AM announcement was scheduled at
-- 05:41 via CronCreate (session-scoped). The session ended before 6:20,
-- the cron job evaporated, and the announcement never fired.
--
-- Fix: every future one-time household action goes through this table.
-- A background worker in benson.service polls + dispatches it.

CREATE TABLE IF NOT EXISTS scheduled_actions (
    id             SERIAL PRIMARY KEY,
    action_type    TEXT NOT NULL,
    action_params  JSONB NOT NULL,
    fire_at        TIMESTAMPTZ NOT NULL,
    created_by     TEXT NOT NULL DEFAULT 'benson',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    fired_at       TIMESTAMPTZ,
    status         TEXT,
    last_error     TEXT
);

CREATE INDEX IF NOT EXISTS idx_scheduled_actions_fire_at
    ON scheduled_actions (fire_at)
    WHERE fired_at IS NULL;
