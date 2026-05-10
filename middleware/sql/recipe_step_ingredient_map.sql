-- recipe_step_ingredient_map.sql — 2026-05-10
--
-- Adds a per-recipe cache of which ingredients are actively used in
-- each step. Computed once via Claude Haiku (see GET
-- /api/recipes/<id>/cook_map in hub.py) and reused on every subsequent
-- /recipes/<id>/cook page load. NULL means "not computed yet".
--
-- Shape: JSONB object mapping a 0-based step index (string key) to an
-- array of 0-based ingredient indices, e.g. {"0": [0,1,3], "1": [2]}.
-- Resolves semantic references like "the dough" / "the sauce" /
-- "remaining mixture" back to their underlying ingredients — something
-- the previous client-side substring matcher could not do.

ALTER TABLE recipes
    ADD COLUMN IF NOT EXISTS step_ingredient_map JSONB DEFAULT NULL;
