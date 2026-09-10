"""One-command diagnostic for Yahoo Fantasy API access.

Access has two independent gates and they fail identically from the outside, so
this separates them: OAuth (do we hold a valid token?) and *authorization scope*
(has the Fantasy API application been approved and the agreement executed?).

A refresh that returns 200 followed by endpoints returning 401
`additional_authorization_required` means the token is fine and the application
is still pending — no amount of re-authenticating fixes that.

Deliberately raw httpx, not yahoo_fantasy_api: this must report what the wire
says without a client library's error handling in the way, and it must keep
working when access is broken. The yahoo_fantasy_api import boundary in
client.py is unaffected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from puckpilot.config import Settings
from puckpilot.yahoo.auth import TOKEN_URL

FANTASY_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"

# Ordered cheapest/most-public first, so the first failure localizes the problem.
PROBES: tuple[tuple[str, str], ...] = (
    ("game metadata", f"{FANTASY_BASE}/game/nhl?format=json"),
    ("my nhl games", f"{FANTASY_BASE}/users;use_login=1/games;game_keys=nhl?format=json"),
    ("my leagues", f"{FANTASY_BASE}/users;use_login=1/games;game_keys=nhl/leagues?format=json"),
)

SCOPE_PENDING = "additional_authorization_required"


@dataclass
class ProbeResult:
    refresh_ok: bool
    refresh_detail: str
    endpoints: list[tuple[str, int, str]]
    scope_granted: bool
    text: str


def _refresh(settings: Settings) -> tuple[bool, str, str | None]:
    token_path = settings.resolved_token_path
    if not token_path.exists():
        return False, f"no token file at {token_path}", None
    tok = json.loads(token_path.read_text())
    if not tok.get("refresh_token"):
        return False, "token file has no refresh_token; re-run the consent flow", None
    resp = httpx.post(
        TOKEN_URL,
        auth=(tok["consumer_key"], tok["consumer_secret"]),
        data={
            "grant_type": "refresh_token",
            "refresh_token": tok["refresh_token"],
            "redirect_uri": "https://localhost:9000",
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}", None
    return True, "refresh token accepted", resp.json()["access_token"]


def probe(settings: Settings | None = None) -> ProbeResult:
    """Check OAuth and Fantasy scope. Never writes the token file."""
    settings = settings or Settings()
    lines: list[str] = ["Yahoo Fantasy API probe", "=" * 40]

    ok, detail, access_token = _refresh(settings)
    lines.append(f"OAuth refresh:  {'OK' if ok else 'FAIL'}  ({detail})")
    if not ok:
        lines += ["", "Fix OAuth first: `ppilot league show` walks the consent flow."]
        return ProbeResult(False, detail, [], False, "\n".join(lines))

    headers = {"Authorization": f"Bearer {access_token}"}
    endpoints: list[tuple[str, int, str]] = []
    lines.append("")
    for name, url in PROBES:
        try:
            r = httpx.get(url, headers=headers, timeout=30.0)
            status, body = r.status_code, r.text
        except httpx.HTTPError as e:
            status, body = 0, str(e)
        note = SCOPE_PENDING if SCOPE_PENDING in body else ("ok" if status == 200 else body[:120])
        endpoints.append((name, status, note))
        lines.append(f"  {name:<16} HTTP {status:<4} {note}")

    granted = any(s == 200 for _, s, _ in endpoints)
    pending = any(n == SCOPE_PENDING for _, _, n in endpoints)
    lines.append("")
    if granted:
        lines.append("VERDICT: Fantasy API access is LIVE.")
    elif pending:
        lines += [
            f"VERDICT: OAuth is fine; Fantasy scope is NOT granted ({SCOPE_PENDING}).",
            "",
            "The token is valid, so re-authenticating will not help on its own.",
            "Either the application/agreement is still being processed by Yahoo, or",
            "access was granted after this token was issued - in which case the",
            "cached grant predates it and a fresh consent IS needed. If Yahoo has",
            "confirmed access, re-run the consent flow; otherwise wait.",
        ]
    else:
        lines.append("VERDICT: unrecognized failure — see the status codes above.")
    return ProbeResult(True, detail, endpoints, granted, "\n".join(lines))
