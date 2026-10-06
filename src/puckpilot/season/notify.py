"""A push to the manager's phone when something needs them.

Week 1's two pickups sat for three days and lapsed, because nothing told
anyone they were there: the page only helps if it is opened. This sends a
notification when a run queues a new pickup to decide on, and when a change
the manager is relying on - tonight's lineup - was not made.

Delivery is any ntfy-compatible endpoint (the free ntfy.sh service and its app,
or a self-hosted server): one HTTP POST with the text as the body. The URL is a
secret - anyone who knows the topic can read it - so it comes from the
environment (`PUCKPILOT_NOTIFY_URL`) and never from config or the repo. Unset,
nothing is sent and nothing fails.

A notification never carries the page key. Its link is the bare page URL; the
page remembers its key once it has been opened with one.
"""

from __future__ import annotations

import os

NOTIFY_ENV = "PUCKPILOT_NOTIFY_URL"


def _header(text: str) -> str:
    # HTTP headers are latin-1; a dash or an accented name must not fail a send.
    return text.encode("latin-1", "replace").decode("latin-1")


def send(
    title: str, body: str, click: str = "", priority: str = "default", tags: str = "", post=None
) -> bool:
    """Send one notification. True if the service took it; never raises."""
    url = os.environ.get(NOTIFY_ENV, "").strip()
    if not url:
        return False
    headers = {"Title": _header(title), "Priority": priority}
    if click:
        headers["Click"] = click
    if tags:
        headers["Tags"] = tags
    try:
        if post is None:
            import httpx

            post = httpx.post
        r = post(url, content=body.encode("utf-8"), headers=headers, timeout=10)
        return 200 <= getattr(r, "status_code", 200) < 300
    except Exception:  # noqa: BLE001 - a missed notification must never fail a run
        return False


def new_pickups(proposals, page_url: str = "", post=None) -> bool:
    """One notification for the pickups a run just queued."""
    if not proposals:
        return False
    lines = []
    for p in proposals:
        gain = p.reason.get("expected_gain")
        head = f"Add {p.add_name}" + (f" for {p.drop_name}" if p.drop_player_key else "")
        worth = f": +{float(gain):.2f} categories this week" if gain is not None else ""
        helps = p.reason.get("helps") or []
        lines.append(head + worth + (f" ({', '.join(helps[:2])})" if helps else ""))
    n = len(proposals)
    title = "New pickup to decide on" if n == 1 else f"{n} pickups to decide on"
    return send(title, "\n".join(lines), click=page_url, tags="ice_hockey", post=post)


def made(what: str, page_url: str = "", post=None) -> bool:
    """An approved move went through: the roster read afterwards shows it."""
    return send(
        f"Done: {what}", "Made in Yahoo.", click=page_url, tags="white_check_mark", post=post
    )


def failed(what: str, why: str, page_url: str = "", post=None) -> bool:
    """Something the manager is relying on did not happen."""
    title = f"NOT done: {what}"
    return send(title, why, click=page_url, priority="high", tags="warning", post=post)
