# Benson — Handoff to Casey

**State as of 2026-04-25, end of autonomous run.**
**Build is functional.** Phases 1-5 are live, Phase 8 (House Hub) is live,
Phases 6-7 are code-complete and waiting on hardware (ReSpeaker satellites)
and tokens (Anthropic, Telegram, Bond, HA, Instacart).

## 🏠 The House Hub is up — visit it now

**`http://Benson.local:8100/`** in any browser on the home network.

Pages: Home (today's meal + chore summary + recent conversations),
Recipes (browse all 58, search, filter by course), Recipe detail,
Weekly Menu (next 7 days), Chores (filter by person, toggle done),
Memory (everything Benson knows), Status (DB + secret state).

Plus a floating **"Talk to Benson" chat widget** on every page.
Try it with "what chores does Zander have today?" or "do we have a
recipe for tortilla soup?" — server-side RAG pulls real DB rows
into the prompt, so answers are grounded, not invented.

Open `/opt/benson/logs/build_log.md` for the full per-phase record. This
file is the **punch list of things only you can do**.

---

## What's running right now

| Service | Port | Status |
|---|---|---|
| postgresql | 5432 | active — `benson` DB has 58 recipes, 26 weekly_plan rows, 405 chores, 16 memories |
| ollama | 11434 | active — llama3.1:70b (42 GB) and llama3.1:8b (4.9 GB) loaded |
| homeassistant | 8123 | active — onboarding wizard waiting on first browser visit |
| benson | 8100 | active — `GET /health` returns ok, `POST /conversation` returns Tier 1 responses end-to-end |

Test the middleware right now:
```
curl -s http://localhost:8100/health
curl -s -X POST http://localhost:8100/conversation \
  -H "Content-Type: application/json" \
  -d '{"text":"how many recipes do we have?","speaker":"Casey","room":"kitchen"}'
```

---

## Things only you can do — in priority order

### 1. Onboard Home Assistant (5 min, browser)
Go to `http://Benson.local:8123` and create the first user. Pick a
strong password. Then **Profile → Long-Lived Access Tokens → Create Token**
named `benson-middleware`. Add the token to `/etc/benson/env` as
`HA_LONG_LIVED_TOKEN`:
```
sudo nano /etc/benson/env
```

### 2. Add the Anthropic API key (1 min)
Get one at https://console.anthropic.com/settings/keys (you have an
account; create a fresh API key — separate from the OAuth token Claude
Code uses). Paste into `/etc/benson/env` as `ANTHROPIC_API_KEY`.

After this is in:
```
sudo systemctl restart benson
```
Then test Tier 2:
```
curl -s --max-time 240 -X POST http://localhost:8100/conversation \
  -H "Content-Type: application/json" \
  -d '{"text":"plan a week of dinners using the recipes we already have","speaker":"Casey","room":"kitchen"}'
```
Expect ~30-60 s (xhigh effort), response summarized through Tier 1 to
stay in Benson's voice.

### 3. Bond bridge token (2 min, physical)
Unplug the Bond bridge, wait 10 s, plug it back in. Within 30 s:
```
curl -s http://192.168.0.132/v2/token
```
Save the returned `token` to `/etc/benson/env` as `BOND_BRIDGE_TOKEN`.
Then in HA UI: Settings → Devices & Services → Add → Bond → host
`192.168.0.132` + token. HA should discover all 4 fans.

### 4. Sonos (1 min, UI)
Optional pre-step: rename the Sonos zone labeled "Bathroom" to
"Master Bedroom" in the Sonos app on your phone, since it physically
lives in the master bedroom.

In HA UI: Settings → Devices & Services → look for the Sonos card →
"Set up". Auto-discovers all 5 zones.

### 5. Telegram bot (5 min)
- @BotFather → `/newbot` → save the HTTP API token to `/etc/benson/env`
  as `TELEGRAM_BOT_TOKEN`.
- Each household member who should be able to message Benson messages
  @userinfobot — note their numeric chat ID. Comma-separate into
  `TELEGRAM_ALLOWED_CHAT_IDS`.
- HA UI: Settings → Devices & Services → Add → "Telegram bot" → polling
  → paste token + chat IDs.
- The HA package YAML I wrote at
  `/opt/benson/ha/.homeassistant/packages/benson_telegram.yaml` will
  auto-route text and TikTok/YouTube/Instagram links to the middleware.
  **Photo and voice handlers are stubbed** — I'll write the helper
  daemon for those when you're back and we can iterate against a real bot.

### 6. Order ReSpeaker hardware (~$176)
4 × ReSpeaker Lite Voice Kit, 4 × 5W speakers (only Cole's room actually
needs the speaker, but order all 4 in case TV Room satellite gets added
later), 4 × USB-C adapters. See `test_scenarios.md` Phase 6 for the
deployment plan; `/opt/benson/context/respeaker/README.md` for the
flashing sequence.

### 7. Apply for Instacart Connect API key (5 min, then ~1 week wait)
https://docs.instacart.com/developer_platform_api/ — request developer
access. Until approved, the `/grocery/instacart` endpoint returns a
fallback search URL.

### 8. Create the kitchen iPad dashboard (5 min, UI)
HA UI → Settings → Dashboards → Add Dashboard → "From YAML" → paste
the contents of `/opt/benson/ha/.homeassistant/lovelace_kitchen_console.yaml`.
Open that dashboard on the kitchen iPad in the HA Companion app and
pin it as the home view.

---

## What's done, that you don't need to touch

- All scaffolding under `/opt/benson/{context,middleware,whisper,piper,openwakeword,scripts,logs,recipes,backups}`.
- Postgres + pgvector, schema, **58 recipes / 26 weekly plans / 405 chores migrated** from miniserver:/opt/recipeapp/recipes.db, 15 seed memories embedded.
- Ollama 70b + 8b downloaded and tested.
- faster-whisper (CPU/int8 — see build_log for the CUDA caveat), Piper TTS with `en_US-lessac-high` voice, openWakeWord with 5 default models.
- HA Core 2025.1.4 in venv, systemd-managed, configuration.yaml + packages dir wired up.
- Middleware (FastAPI on :8100) with all endpoints: /health, /conversation, /recipe/photo, /recipe/video, /grocery/instacart, /memory/search, /memory/store. Memory auto-extraction working.
- Benson system prompt loaded; tested live with one Tier 1 conversation that produced a properly-toned response and auto-stored a memory.
- Per-satellite HA routing YAML, ESPHome satellite template, kitchen iPad Lovelace dashboard, wake-word training docs — all written and waiting for hardware.

---

## Caveats worth your attention

1. **Whisper is on CPU**, not GPU — `ctranslate2` PyPI wheel for arm64
   doesn't bundle CUDA support. CPU large-v3 transcribes at ~5× realtime
   (a 10 s voice message takes ~50 s). Acceptable for async Telegram
   voice; suboptimal for live satellite STT. Mitigations listed in the
   Phase 2 build_log entry. Recommend: ship as-is, swap in
   `distil-large-v3` (3× faster, similar accuracy) only if it becomes
   noticeable.

2. **Tier 1 (Ollama 70b) cold-load is ~47 s.** Once warm it's ~6 tok/s,
   so a 60-token spoken response is ~10 s. The model stays warm as long
   as queries keep arriving; idle eviction has been Ollama's default.
   If responses feel cold-laggy in normal use, we can pin the model
   resident with `OLLAMA_KEEP_ALIVE=24h` in the systemd unit.

3. **Chores has a third "person" you didn't mention: `General` (81
   rows).** Things like "refinish decks," "trim scrub oak," "wash
   windows" — household-wide tasks. Worth deciding: should kid-tier
   queries include these in "what chores are due today" answers? My
   guess: no — only their own. But it's your call. Until then the
   middleware doesn't differentiate; it'll show all chores when asked.

4. **NOPASSWD sudo is still active.** Revoke when you're done with the
   build:
   ```
   sudo rm /etc/sudoers.d/benson-build
   ```
   Also: rotate the sudo password you pasted earlier in the session.

5. **Two empty source slots** under `/opt/benson/context/prior_code/`
   (Alfred and todo_pipeline) — Casey skipped them. Leaving the empty
   directories so they're easy to fill if material surfaces later.

---

## Files of interest

- `/opt/benson/logs/build_log.md` — full per-phase record.
- `/opt/benson/context/00_README.md` — mission/scope/refusals (load-bearing).
- `/opt/benson/context/HA_INTEGRATIONS_TODO.md` — UI integration steps.
- `/opt/benson/context/secrets_checklist.md` — per-key acquisition guide.
- `/opt/benson/context/tests/test_scenarios.md` — full validation gate.
- `/opt/benson/context/respeaker/README.md` — satellite deployment.
- `/etc/benson/env` — secrets (root:root 600).

Tail logs:
```
sudo journalctl -u benson -f
sudo journalctl -u homeassistant -f
sudo journalctl -u ollama -f
```

Welcome back.
