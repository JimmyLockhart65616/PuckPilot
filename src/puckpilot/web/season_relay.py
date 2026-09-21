"""Serve the in-season view, and carry decisions back.

Same shape as the draft relay and the same reason for it: the deciding happens
where the database and the Yahoo session are, which is somebody's PC, and what
moves to a public host is only the view. This one adds a return path, because
a transaction needs a person to approve it and that person is holding a phone.

Two things differ from the draft relay, and both follow from a season being
194 days rather than one evening.

**It is keyed by manager, not by seat.** Two managers in the same league are
rivals; a key admits you to your own view and to nothing else. There is no
guest role, because there is no second person who should see a manager's
waiver plan.

**Decisions are collected, not stored.** A decision lands here and waits for
the local job to drain it. If this process restarts first, the decision is
lost - and the consequence of that is the proposal reappearing on the next
push for you to decide again. That is the direction the failure has to fall:
losing an approval costs a tap, while inventing one spends an acquisition. So
nothing here is durable on purpose, and the local side treats a drained
decision as the only decision.

No engine imports; this file is copied into a standard-library-only image.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from puckpilot.web import wire
from puckpilot.web.season_page import PAGE

MAX_PUSH_BYTES = 4 * 1024 * 1024

# A lineup that was right this morning is wrong by evening, so the page says
# how old it is and goes loud past this. Generous compared with the draft's 30
# seconds because the in-season job runs on a schedule, not continuously.
STALE_AFTER_S = 3 * 3600.0

# More than this waiting to be drained means nobody is collecting them, which
# is worth failing loudly about rather than growing without bound.
MAX_PENDING_DECISIONS = 200

BUILD_FILES = ("season_page.py", "season_relay.py", "wire.py")


def build_id(here: Path | None = None) -> str:
    """Short content hash of what this relay serves.

    Line endings are normalised first: a Windows checkout and the Linux image
    hold the same code with different bytes, and that is not drift.
    """
    h = hashlib.sha256()
    here = here or Path(__file__).resolve().parent
    for name in BUILD_FILES:
        h.update(name.encode())
        h.update((here / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def parse_keys(spec: str) -> dict[str, str]:
    """`jimmy:abc,sam:def` -> {key: manager}.

    Keyed by the secret rather than the name so a lookup is a single constant
    time comparison per manager and never leaks which names exist.
    """
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, key = part.split(":", 1)
        name, key = name.strip(), key.strip()
        if name and key:
            out[key] = name
    return out


class SeasonState:
    """The last snapshot per manager, and the decisions waiting to be collected."""

    def __init__(self, stale_after: float = STALE_AFTER_S):
        self._lock = threading.Lock()
        self._snap: dict[str, dict] = {}
        self._at: dict[str, float] = {}
        self._decisions: list[dict] = []
        self._seq = 0
        self.stale_after = stale_after
        self.started = time.time()

    # -- the view ----------------------------------------------------------

    def push(self, manager: str, snapshot: dict) -> None:
        with self._lock:
            self._snap[manager] = snapshot
            self._at[manager] = time.time()

    def get(self, manager: str) -> dict:
        with self._lock:
            snap = self._snap.get(manager)
            at = self._at.get(manager)
        if snap is None:
            # A cold relay must render as a page that says so, not as a page
            # missing half its keys - the draft build learned that the hard way
            # when a fresh URL drew "seat undefined".
            return {
                "empty": True,
                "manager": manager,
                "stale": True,
                "age_seconds": None,
                "moves": [],
                "proposals": [],
                "roster": [],
                "week": None,
                "protocol": None,
            }
        age = time.time() - at
        out = dict(snap)
        out["manager"] = manager
        out["empty"] = False
        out["age_seconds"] = round(age, 1)
        out["stale"] = age > self.stale_after
        return out

    # -- the return path ---------------------------------------------------

    def decide(self, manager: str, kind: str, ident: int, approve: bool) -> dict:
        if kind not in ("proposal", "protocol"):
            raise ValueError(f"unknown decision kind {kind!r}")
        with self._lock:
            if len(self._decisions) >= MAX_PENDING_DECISIONS:
                raise RuntimeError("too many undelivered decisions; is the local job running?")
            self._seq += 1
            row = {
                "seq": self._seq,
                "manager": manager,
                "kind": kind,
                "id": int(ident),
                "approve": bool(approve),
                "at": time.time(),
            }
            self._decisions.append(row)
            # Reflect it immediately so the page does not show a decided item
            # as still pending until the next push.
            self._mark_decided(manager, kind, int(ident), bool(approve))
        return row

    def _mark_decided(self, manager: str, kind: str, ident: int, approve: bool) -> None:
        snap = self._snap.get(manager)
        if not snap:
            return
        verdict = "approved" if approve else "rejected"
        if kind == "protocol":
            p = snap.get("protocol")
            if isinstance(p, dict) and p.get("id") == ident:
                p["status"] = verdict
        else:
            snap["proposals"] = [p for p in snap.get("proposals", []) if p.get("id") != ident]

    def drain(self, manager: str) -> list[dict]:
        """Hand over this manager's decisions exactly once."""
        with self._lock:
            mine = [d for d in self._decisions if d["manager"] == manager]
            self._decisions = [d for d in self._decisions if d["manager"] != manager]
        return mine

    def health(self) -> dict:
        with self._lock:
            managers = sorted(self._snap)
            waiting = len(self._decisions)
        return {
            "ok": True,
            "build": build_id(),
            "git": os.environ.get("PUCKPILOT_GIT_SHA", ""),
            "uptime_s": round(time.time() - self.started, 1),
            "managers": len(managers),
            "decisions_waiting": waiting,
        }


def make_handler(state: SeasonState, keys: dict[str, str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _refuse(self, code, msg):
            self._send(code, wire.dumps({"error": msg}), "application/json")

        def _route(self) -> str:
            return self.path.split("?", 1)[0].rstrip("/") or "/"

        def _manager(self) -> str | None:
            given = (
                self.headers.get("X-PuckPilot-Key")
                or (parse_qs(urlparse(self.path).query, keep_blank_values=True).get("k") or [""])[0]
            )
            # compare_digest raises on non-ASCII, and the key arrives in a query
            # string anyone can type. A wrong key has to be a 403, not a 500.
            if not given or not given.isascii():
                return None
            for key, manager in keys.items():
                if secrets.compare_digest(given, key):
                    return manager
            return None

        def do_GET(self):
            route = self._route()
            if route == "/healthz":
                self._send(200, wire.dumps(state.health()), "application/json")
                return
            manager = self._manager()
            if manager is None:
                if route in ("/state", "/decisions"):
                    self._refuse(403, "a valid access key is required")
                else:
                    # Serve the page anyway: it renders its own "wrong key"
                    # message, which beats a bare 403 for someone on a phone.
                    self._send(200, PAGE, "text/html; charset=utf-8")
                return
            if route == "/state":
                self._send(200, wire.dumps(state.get(manager)), "application/json")
            elif route == "/decisions":
                self._send(200, wire.dumps({"decisions": state.drain(manager)}), "application/json")
            else:
                self._send(200, PAGE, "text/html; charset=utf-8")

        def _body(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._refuse(400, "bad Content-Length")
                return None
            if length <= 0 or length > MAX_PUSH_BYTES:
                self._refuse(413, "body missing or too large")
                return None
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except ValueError as e:
                self._refuse(400, f"bad body: {e}")
                return None

        def do_POST(self):
            manager = self._manager()
            if manager is None:
                self._refuse(403, "a valid access key is required")
                return
            route = self._route()
            body = self._body()
            if body is None:
                return
            if route == "/push":
                snap = body.get("snapshot")
                if not isinstance(snap, dict):
                    self._refuse(400, "push needs a snapshot object")
                    return
                # Cleaned on the way in as well as out: a client older than the
                # strict wire can still send NaN, and JSON.parse rejects it.
                state.push(manager, wire.clean(snap))
                self._send(200, wire.dumps({"ok": True, "manager": manager}), "application/json")
            elif route == "/decide":
                try:
                    row = state.decide(
                        manager,
                        str(body.get("kind")),
                        int(body.get("id")),
                        bool(body.get("approve")),
                    )
                except (TypeError, ValueError) as e:
                    self._refuse(400, str(e))
                except RuntimeError as e:
                    self._refuse(503, str(e))
                else:
                    self._send(200, wire.dumps({"ok": True, "seq": row["seq"]}), "application/json")
            else:
                self._refuse(404, "not found")

    return Handler


def serve(
    state: SeasonState,
    keys: dict[str, str],
    port: int,
    host: str = "0.0.0.0",  # noqa: S104
) -> ThreadingHTTPServer:
    """Bind and start. Public by construction - the keys are the guard."""
    server = ThreadingHTTPServer((host, port), make_handler(state, keys))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    """Container entry point. Keys come from the environment, never a literal."""
    keys = parse_keys(os.environ.get("PUCKPILOT_MANAGER_KEYS", ""))
    if not keys:
        print(
            "PUCKPILOT_MANAGER_KEYS must be set, as 'name:key,name:key'",
            flush=True,
        )
        return 2
    port = int(os.environ.get("PORT", "8080"))
    serve(SeasonState(), keys, port)
    print(f"season relay listening on :{port} for {len(keys)} manager(s)", flush=True)
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
