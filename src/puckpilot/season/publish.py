"""Pushing the view out and collecting decisions back.

Both directions are deliberately failure-tolerant. A dead relay must not end a
lineup run - the decision is still correct and still printed - and a decision
that cannot be applied must not stop the others. The draft console learned the
first of these the hard way: its push was built outside its own guard, so a URL
without a scheme ended the draft rather than the push.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

TIMEOUT_S = 10.0


class PublishError(RuntimeError):
    """The relay could not be reached or refused us."""


def _call(url: str, key: str, body: dict | None = None) -> dict:
    from puckpilot.web import wire

    data = wire.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"X-PuckPilot-Key": key, "Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        raise PublishError(f"{e.code} from the relay: {detail}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise PublishError(f"could not reach the relay: {e}") from e


def push(url: str, key: str, snapshot: dict) -> dict:
    return _call(url.rstrip("/") + "/push", key, {"snapshot": snapshot})


def collect(url: str, key: str) -> list[dict]:
    """Drain this manager's decisions. They are handed over exactly once."""
    out = _call(url.rstrip("/") + "/decisions", key)
    got = out.get("decisions")
    return got if isinstance(got, list) else []


def health(url: str) -> dict:
    return _call(url.rstrip("/") + "/healthz", "")
