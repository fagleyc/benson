-- chores_recurring.sql — 2026-05-01
--
-- Adds optional `recurring` column to chores. NULL means one-off (current
-- behavior unchanged). When set, the nightly rollover job (in
-- nightly_index.py) regenerates a fresh undone copy on the next applicable
-- day. Independent of the rollover-undone-chores behavior, which moves
-- ANY incomplete chore (recurring or not) forward so it doesn't vanish
-- from "today's" view.
--
-- Allowed values: 'daily' | 'weekly' | 'weekdays' | 'weekends'

ALTER TABLE chores
    ADD COLUMN IF NOT EXISTS recurring TEXT DEFAULT NULL
    CHECK (recurring IS NULL
           OR recurring IN ('daily','weekly','weekdays','weekends'));
