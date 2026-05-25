-- Homebrew Pandora: user-defined "station" seed cards.
--
-- A station bundles genre / decade / mood / seed-artist preferences. When
-- played, the music handler asks Music Assistant to search for tracks
-- matching the bundle and queues them onto the selected Sonos zone.
--
-- One row per saved station. Seeded with 7 defaults on first boot.

CREATE TABLE IF NOT EXISTS music_stations (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    seeds           JSONB NOT NULL,
    -- {genres: [...], decades: [...], moods: [...],
    --  seed_artists: [...], seed_tracks: [...]}
    cover_palette   JSONB,            -- {hex_from, hex_to} for the card gradient
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_played_at  TIMESTAMPTZ,
    play_count      INT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_music_stations_last_played
    ON music_stations (last_played_at DESC NULLS LAST);
