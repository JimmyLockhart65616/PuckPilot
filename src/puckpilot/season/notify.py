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
    """One notification for the pickups a run just queued.

    A make-room move leads, at high priority: until it is made the roster is
    illegal and no pickup in the same message can be made either.
    """
    if not proposals:
        return False
    from puckpilot.season.moves import kind_of

    rooms = [p for p in proposals if not kind_of(getattr(p, "kind", None)).spends_acquisition]
    proposals = [p for p in proposals if p not in rooms]
    lines = [
        kind_of(p.kind).head(p.add_name, p.drop_name, str(p.reason.get("slot", "")))
        + " - no add can be made until it is"
        for p in rooms
    ]
    for p in proposals:
        gain = p.reason.get("expected_gain")
        head = f"Add {p.add_name}" + (f" for {p.drop_name}" if p.drop_player_key else "")
        worth = f": +{float(gain):.2f} categories this week" if gain is not None else ""
        helps = p.reason.get("helps") or []
        lines.append(head + worth + (f" ({', '.join(helps[:2])})" if helps else ""))
    n = len(proposals)
    title = "New pickup to decide on" if n == 1 else f"{n} pickups to decide on"
    if rooms:
        title = "Roster blocked: approve the make-room move" + (f" (+{n} pickups)" if n else "")
        return send(
            title, "\n".join(lines), click=page_url, priority="high", tags="warning", post=post
        )
    return send(title, "\n".join(lines), click=page_url, tags="ice_hockey", post=post)


def standing_offers(items, page_url: str = "", post=None) -> bool:
    """Moves PuckPilot makes under the manager's standing approval unless he
    stops them, each with its time - the window is what makes it his yes, so
    the push says when. `items` is (proposal, "9:05 PM") pairs."""
    if not items:
        return False
    from puckpilot.season.moves import kind_of

    lines = [
        kind_of(getattr(p, "kind", None)).head(
            p.add_name, p.drop_name if p.drop_player_key else "", str(p.reason.get("slot", ""))
        )
        + f" - at {when} unless you stop it"
        for p, when in items
    ]
    first = items[0][1]
    title = (
        f"PuckPilot makes this at {first} unless you stop it"
        if len(items) == 1
        else f"{len(items)} moves PuckPilot makes unless you stop them"
    )
    return send(title, "\n".join(lines), click=page_url, tags="hourglass", post=post)


def made(what: str, page_url: str = "", post=None) -> bool:
    """An approved move went through: the roster read afterwards shows it."""
    return send(
        f"Done: {what}", "Made in Yahoo.", click=page_url, tags="white_check_mark", post=post
    )


def ask_again(what: str, page_url: str = "", post=None) -> bool:
    """An approval given on "make it yourself", back on the page for one more
    tap now that PuckPilot makes the move - a card that reappears silently is
    one nobody looks for."""
    return send(
        f"One more tap: {what}",
        "Approved when the card said to make it yourself. PuckPilot makes it now - "
        "approve it again on the page.",
        click=page_url,
        tags="point_right",
        post=post,
    )


def blocked(title: str, message: str, page_url: str = "", post=None) -> bool:
    """Something stops the tool working (`season/blockers.py`) - once a day
    while it stands, at high priority."""
    return send(
        f"Blocked: {title}", message, click=page_url, priority="high", tags="rotating_light",
        post=post,
    )  # fmt: skip


def failed(what: str, why: str, page_url: str = "", post=None) -> bool:
    """Something the manager is relying on did not happen."""
    title = f"NOT done: {what}"
    return send(title, why, click=page_url, priority="high", tags="warning", post=post)
