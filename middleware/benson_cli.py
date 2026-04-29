#!/opt/benson/middleware/venv/bin/python
"""benson — terminal client for the Benson household AI.

Examples:
    benson "what time is it"
    benson "add laundry to Cole's chores for tomorrow"
    benson --speaker Lindsey --room kitchen "what's for dinner"
    benson --voice --room patio "introduce yourself to the family"
    echo "remember that Lindsey doesn't eat shellfish" | benson
    benson --json "how many recipes do we have"

Reads ANTHROPIC_API_KEY etc. from /etc/benson/env (already loaded by the
benson.service systemd unit). Talks to the local middleware over HTTP, so
authentication is by virtue of being on the host (or LAN).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("BENSON_URL", "http://localhost:8100")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="benson",
        description="Talk to the Benson household AI from the command line.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="What to ask. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--speaker", "-s",
        default=os.environ.get("USER", "Casey").capitalize(),
        help="Who is speaking (default: $USER capitalized).",
    )
    parser.add_argument(
        "--room", "-r",
        default="cli",
        help="Where the request originates. Voice replies route by room. "
             "Default: 'cli' (text-only, no spoken reply).",
    )
    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        help="Also speak the response through the room's Sonos zone.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON response instead of just the text.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Middleware base URL (default: {DEFAULT_URL}).",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--forget",
        action="store_true",
        help="Clear the running session for this speaker/room and exit.",
    )
    args = parser.parse_args()

    if args.forget:
        return _forget(args.url, args.speaker, args.room)

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        if sys.stdin.isatty():
            parser.error("no prompt provided (and no stdin)")
        prompt = sys.stdin.read().strip()
        if not prompt:
            parser.error("empty stdin")

    payload = {
        "text": prompt,
        "speaker": args.speaker,
        "room": args.room,
        "voice_input": args.voice,
    }
    req = urllib.request.Request(
        args.url.rstrip("/") + "/conversation",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"error: HTTP {e.code}\n{e.read().decode(errors='replace')}\n")
        return 1
    except urllib.error.URLError as e:
        sys.stderr.write(f"error: cannot reach Benson at {args.url}: {e.reason}\n")
        return 2

    data = json.loads(body)
    if args.json:
        json.dump(data, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(data.get("response", "").rstrip())
    meta_bits = [data.get("tier", "?")]
    if data.get("spoken_on"):
        meta_bits.append(f"spoken on {data['spoken_on']}")
    sys.stderr.write(f"[{' · '.join(meta_bits)}]\n")
    return 0


def _forget(url: str, speaker: str, room: str) -> int:
    req = urllib.request.Request(
        url.rstrip("/") + "/agent/forget",
        data=json.dumps({"speaker": speaker, "room": room}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    print("session cleared" if data.get("cleared") else "no active session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
