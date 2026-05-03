#!/usr/bin/env python3
"""Sunday evening chore report.

Tallies the week's chores per kid and posts a structured summary to
Casey via Signal. Designed to land at 7pm Sunday so Casey can act on
it before the kids' bedtime.

Run by: systemd timer `benson-sunday-chores.timer`.
Manual:  `sudo -u casey python3 sunday_chore_report.py`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, "/opt/benson/middleware")

# systemd already sets env from /etc/benson/env via the unit's
# EnvironmentFile directive, so we only do the manual load when the
# script is invoked outside systemd (interactive testing). Skip
# silently on PermissionError — that's the expected case under the
# `casey` user since the env file is root-only.
ENV_FILE = Path("/etc/benson/env")
if ENV_FILE.exists():
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    except PermissionError:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benson.sunday_chores")


def _signal_recipient() -> str | None:
    explicit = (os.environ.get("SIGNAL_REVIEW_RECIPIENT") or "").strip()
    if explicit:
        return explicit
    allowed = (os.environ.get("SIGNAL_ALLOWED_NUMBERS") or "").strip()
    if allowed:
        return allowed.split(",")[0].strip()
    return None


def _format_report(summary_rows: list[dict], items: list[dict],
                   monday: date, sunday: date) -> str:
    lines = [
        f"Weekly chore report — {monday.strftime('%b %d')} to {sunday.strftime('%b %d')}",
        "",
    ]
    by_person = {r["person"]: r for r in summary_rows}

    for kid in ("Cole", "Zander", "General"):
        r = by_person.get(kid)
        if not r:
            continue
        if kid == "Cole":
            earned = float(r.get("earned_dollars") or 0)
            possible = float(r.get("possible_dollars") or 0)
            unit = "$"
            tally = f"${earned:.2f} of ${possible:.2f}"
        elif kid == "Zander":
            earned = int(r.get("earned_points") or 0)
            possible = int(r.get("possible_points") or 0)
            unit = "pts"
            tally = f"{earned} of {possible} pts"
        else:
            continue
        done = r.get("done_count") or 0
        total = r.get("total_count") or 0
        pct = f" ({100*done//total}%)" if total else ""
        lines.append(f"{kid}: {tally} · {done}/{total} chores{pct}")

        # Top miss(es) — undone chores with the largest reward.
        person_items = [
            i for i in items
            if i.get("person") == kid and not i.get("done")
        ]
        person_items.sort(
            key=lambda x: (
                float(x.get("dollars") or 0) if kid == "Cole"
                else int(x.get("points") or 0)
            ),
            reverse=True,
        )
        misses = person_items[:3]
        if misses:
            lines.append("  Missed:")
            for m in misses:
                d = m.get("chore_date")
                d_str = d.strftime("%a") if hasattr(d, "strftime") else str(d)
                if kid == "Cole":
                    val = f"${float(m.get('dollars') or 0):.2f}"
                else:
                    val = f"{int(m.get('points') or 0)}pts"
                lines.append(f"    • {d_str} — {m.get('chore_name')} ({val})")
        lines.append("")

    if len(lines) <= 2:
        lines.append("(no chores tracked this week)")
    lines.append("Pay Cole in Chase First Banking; Zander gets points logged.")
    return "\n".join(lines)


async def main() -> int:
    from agent_tools import weekly_chore_summary
    from signal_handler import send_signal_message

    today = date.today()
    # Run on Sunday — anchor to today's Monday so the week we're closing
    # is Mon..Sun. If invoked off-schedule, anchor to the current week.
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)

    result = await weekly_chore_summary(week_start=monday.isoformat())
    if not result.get("ok"):
        log.warning(f"summary failed: {result}")
        return 1

    msg = _format_report(
        result["summary"], result["items"], monday, sunday
    )

    recipient = _signal_recipient()
    if not recipient:
        log.warning("no SIGNAL_REVIEW_RECIPIENT or SIGNAL_ALLOWED_NUMBERS set; printing")
        print(msg)
        return 0

    log.info(f"sending Sunday chore report to {recipient}")
    r = await send_signal_message(recipient, msg)
    if not r.get("ok"):
        log.warning(f"send failed: {r}")
        return 2
    log.info("delivered")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
