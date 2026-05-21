# Google OAuth verification — submission prep

Goal: get Benson's OAuth app verified so refresh tokens last "until
revoked" instead of being rotated by Google's unverified-app policy
(currently every ~30 days for Casey). Once verified, the monthly
re-link friction disappears entirely.

## Estimated timeline + cost
- Filing the form: ~2 hours of writing + 1 short demo video recording.
- Google review: 4–6 weeks typical for sensitive scopes (gmail.readonly,
  calendar.events). Restricted scopes are 8–12 weeks. We are NOT using
  restricted scopes, so 4–6 weeks is the realistic bracket.
- Cost: $0 for personal use. Google does NOT require a CASA security
  assessment unless you use restricted scopes (Drive, full Gmail, etc.).
- Engineering work post-approval: zero. The same client ID + same code
  keeps running, refresh tokens just stop rotating.

## What's needed (the form will ask for these)

### 1. App homepage URL
- Required: a publicly reachable HTTPS page describing the app, who runs
  it, and what it does. Doesn't need to be fancy — a single GitHub Pages
  page is fine.
- Suggested URL: `https://fagleyc.github.io/benson/` (just point a
  `gh-pages` branch at a one-page site, or use a `docs/index.html` with
  Pages enabled on `main`).
- Content (single page is enough):
  - Title: "Benson — Fagley household AI assistant"
  - One paragraph: what Benson is (a private home-automation AI used by
    one family on local-network-only hardware in Colorado Springs).
  - One paragraph: who uses it (Casey Fagley + family members on linked
    Google accounts — no public sign-up, no third-party access).
  - Contact: `caseyfagley@gmail.com`.

### 2. Privacy policy URL
- Required: also publicly reachable HTTPS. Google reads it.
- Suggested location: same GitHub Pages site, path `/privacy`.
- Required content (Google's checklist):
  - The exact app name ("Benson").
  - The data types collected (which scopes — list each: gmail.readonly,
    calendar.events, calendar.readonly, openid, userinfo.email,
    userinfo.profile).
  - How the data is used (per-scope: e.g., "Gmail read scope is used to
    let the user ask Benson to search or summarize their own inbox via
    natural language. No emails are stored, indexed, retained, or
    transmitted beyond the user's home network.").
  - How long data is retained ("transient — emails fetched per-query
    are never persisted; calendar data is cached for 15 min then
    discarded").
  - Whether data is shared with third parties ("no, Benson runs entirely
    on local hardware in the user's home; no analytics, no telemetry").
  - User rights (revoke access at any time via
    `myaccount.google.com/permissions`).
  - Contact email.
  - Last-updated date.

  Don't copy-paste a generic template — Google's reviewers do read it
  and reject vague language. Mention "local-only", "no cloud storage",
  "no third-party sharing" explicitly.

### 3. Scope justifications (per scope)
You'll paste one paragraph per scope. Draft text below — refine for the
form, but the substance is correct.

**`https://www.googleapis.com/auth/gmail.readonly`**
> Benson lets household members ask natural-language questions about
> their own inbox via voice or chat ("search recent email from the
> insurance agent", "did Cole's school send anything today"). All
> queries run against the authenticated user's own messages. Benson
> does not send mail, does not index or store mail server-side, does
> not transmit mail outside the user's home network, and does not
> share mail with other household members — each user's OAuth grant
> is scoped to their own messages only. The app runs on a single
> private LAN-only server (DGX Spark) with no public exposure.

**`https://www.googleapis.com/auth/calendar.events`** + **`calendar.readonly`**
> Benson reads household calendars to answer scheduling questions
> ("what's on the calendar today", "when's Cole's next karate") and
> writes events when a household member explicitly asks Benson to add
> one ("add a dentist appointment for Lindsey next Friday at 3 PM").
> Writes always confirm the user_name and the event details aloud
> before committing. Calendar data is cached in RAM for at most 15
> minutes; nothing is persisted. Per-user OAuth grants ensure each
> household member can only see + modify their own calendar.

**`openid`, `userinfo.email`, `userinfo.profile`**
> Used only to display the linked Google account's email + name on the
> household admin page so the user can confirm they linked the right
> account. Not transmitted, not stored beyond the linked-token record.

### 4. Demo video (YouTube, unlisted)
- Length: 90 seconds is plenty. Google requires "less than 3 minutes".
- Script (record once on phone, hold steady):
  1. (5s) Show the Benson admin page in a browser at
     `https://benson.local/admin/google`. Show the "Connect Google"
     button.
  2. (15s) Click Connect → show Google's standard consent screen → grant
     access. The unverified-app warning will appear — that's fine to
     show; it goes away once Google approves this submission.
  3. (15s) Back at Benson, show the green "linked" status for the user.
  4. (15s) Open the conversation page, type "search recent email from
     Verizon" → show the Gmail-search response.
  5. (15s) Type "add a dentist appointment for next Friday at 3 PM" →
     show the calendar-event confirmation.
  6. (15s) Open `myaccount.google.com/permissions`, show Benson listed
     under "third-party apps with access", click Remove → show Benson's
     admin page reflecting the disconnect.
- Upload to YouTube as "Unlisted". Paste the URL into the form.

### 5. Authorized domains
- `fagleyc.github.io` (homepage + privacy policy)
- Plus whatever domain hosts the OAuth redirect URI. Currently the redirect
  is probably `https://192.168.0.240/...` or `https://benson.local/...`.
  Google won't accept private IPs or .local domains as authorized
  domains. Two options:
    a. Add a real domain (e.g., `benson.fagley.example`) with DNS that
       resolves to 192.168.0.240 inside the LAN. Doesn't need to be
       public-resolvable, but Google does check authorized-domain
       ownership via Search Console verification.
    b. Use `localhost`-style installed-app OAuth flow (different
       client type — "Desktop app" instead of "Web app"). Avoids the
       authorized-domain check entirely. Worth investigating before
       submitting if option (a) is annoying.

### 6. Where to file
- https://console.cloud.google.com/apis/credentials/consent
- Select your existing Benson OAuth project. Hit "Submit for verification".
- Google will email Casey within ~24 hours with a ticket ID and either
  questions (most common — they always have at least one round of
  questions) or "approved" (rare on first round).

## Pre-flight checklist (do these BEFORE clicking Submit)

- [ ] App homepage live + reachable on the open internet (curl from a
      machine NOT on your LAN)
- [ ] Privacy policy live + reachable + last-updated date is recent
- [ ] YouTube demo video uploaded, set to Unlisted, URL works incognito
- [ ] OAuth consent screen "App information" tab fully filled in
      (logo optional but helps; 120×120 px PNG)
- [ ] All test users + Casey + Lindsey listed under "Test users" still
      (until verification completes, only listed test users can grant
      access — make sure family stays on the list during the 4-6 weeks
      of review)
- [ ] Privacy policy mentions every scope by URL string

## During Google's review

- Expect 1–2 rounds of clarifying questions over email. Common ones:
  > "Your scope justification mentions 'household members'. Does this
  > mean the app is multi-user? If so, please describe the user model
  > and how each user's data is isolated."
- Reply within 1–2 days; long delays cause Google to close tickets.
- If they ask for additional security measures (e.g., "data is encrypted
  at rest"), say so honestly: the box is locked down, no public
  exposure, single-tenant. They usually accept that for personal /
  household / small-team scopes.

## What changes after approval

- No code change. Same client ID, same secret, same scopes.
- The unverified-app warning ("Google hasn't verified this app") goes
  away on the consent screen — users see a normal Google sign-in flow.
- Refresh tokens stop rotating on the policy clock. They last until:
  - User revokes via myaccount.google.com/permissions, OR
  - Token unused for 6+ months, OR
  - Account password changed (rare; usually survives).
- The nudge automation (`google_handler._send_oauth_nudge`) becomes a
  silent safety net rather than a monthly reminder.

## If verification is rejected

- Most common reason: insufficient scope justification. Solution: more
  specific use-case description, less generic language.
- Google usually gives concrete feedback and lets you re-submit without
  starting from scratch.
- Final fallback: drop `gmail.readonly` if it turns out Casey rarely
  uses email queries — calendar-only is a much lower bar (still
  "sensitive" but reviewers are softer on it).
