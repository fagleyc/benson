-- Per-track thumbs up/down events for the homebrew Pandora.
--
-- One row per (station, track) pairing. "Latest thumb wins" semantics via
-- the unique partial index — a -1 followed by a +1 will UPSERT the same
-- row, flipping the value. NULL station_id is allowed (global thumbs on
-- ad-hoc playback), COALESCE'd into the unique key.
--
-- Used by /api/music/stations/{id}/fitness and the queue builder to filter
-- thumb=-1 artists/tracks out of upcoming radio play.

CREATE TABLE IF NOT EXISTS music_thumbs (
    id            SERIAL PRIMARY KEY,
    station_id    INT REFERENCES music_stations(id) ON DELETE CASCADE,
    media_uri     TEXT,
    artist        TEXT,
    album         TEXT,
    title         TEXT,
    thumb         SMALLINT NOT NULL CHECK (thumb IN (-1, 0, 1)),
    source        TEXT NOT NULL DEFAULT 'manual',   -- 'manual' | 'skip' | 'completed'
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    zone          TEXT
);
CREATE INDEX IF NOT EXISTS idx_music_thumbs_station ON music_thumbs (station_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_music_thumbs_artist ON music_thumbs (artist) WHERE thumb = -1;
CREATE UNIQUE INDEX IF NOT EXISTS uniq_music_thumbs_latest
  ON music_thumbs (COALESCE(station_id, 0), COALESCE(media_uri, artist || '|' || title));
