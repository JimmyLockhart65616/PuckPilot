"""Serve the draft view to a second manager, from somewhere that is not my PC.

The feed cannot move. It reads the Yahoo draft room's own websocket through a
logged-in browser on the drafter's machine, so the board is necessarily computed
there. What *can* move is the view: this holds the last snapshot the console
pushed and hands it to whoever opens the link.

So it is deliberately dumb. It owns no board, ranks nothing, and imports no part
of the engine - which is why it starts in milliseconds on a slim base image
instead of dragging numpy, pandas and scipy into a container that would never
call them.

State is in memory on purpose: a draft is one evening, and a board that outlives
it is a board that can go stale without saying so. Two consequences for the
deployment, both load-bearing - run exactly one replica, or pushes land on one
instance while reads hit another; and do not scale to zero mid-draft, or the
board is silently gone.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from puckpilot.web.access import Access
from puckpilot.web.page import PAGE

MAX_PUSH_BYTES = 8 * 1024 * 1024


class RelayState:
    """The last snapshot per seat, and when it arrived."""

    def __init__(self) -> None:
        self.seats: dict[str, dict] = {}
        self.pushed_at: float | None = None
        self.pushes = 0
        self._lock = threading.Lock()

    def push(self, seats: dict) -> int:
        with self._lock:
            for seat, snap in seats.items():
                self.seats[str(seat)] = snap
            self.pushed_at = time.time()
            self.pushes += 1
            return len(self.seats)

    def get(self, seat: str | None) -> dict:
        with self._lock:
            if not self.seats:
                return {
                    "waiting": True,
                    "note": "Waiting for the draft console to connect.",
                    "shortlist": [],
                    "board": [],
                    "roster": [],
                    "needs": [],
                    "made": 0,
                    "total": 0,
                    "detected": 0,
                    "n_left": 0,
                    "gaps": [],
                    "seconds_since_pick": None,
                    "diagnostics": "no snapshot pushed yet",
                }
            if seat is None or seat not in self.seats:
                # One seat pushed and no seat asked for: show it rather than
                # erroring, so a bare link still works.
                seat = next(iter(self.seats)) if seat is None else seat
            snap = self.seats.get(seat)
            if snap is None:
                return {"error": f"no snapshot for seat {seat}", "seats": sorted(self.seats)}
            out = dict(snap)
            age = None if self.pushed_at is None else time.time() - self.pushed_at
            out["relay_age"] = age
            return out


def make_handler(state: RelayState, access: Access):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _refuse(self, code, msg):
            self._send(code, json.dumps({"error": msg}), "application/json")

        def _route(self) -> str:
            return self.path.split("?", 1)[0].rstrip("/") or "/"

        def _token(self) -> str:
            header = self.headers.get("X-PuckPilot-Key")
            if header:
                return header
            return (parse_qs(urlparse(self.path).query, keep_blank_values=True).get("k") or [""])[0]

        def _role(self) -> str | None:
            given = self._token()
            # compare_digest raises on non-ASCII str, and the key arrives in a
            # query string anyone can type. A crash is a 500 with a traceback in
            # the log; a wrong key is a 403. It has to be the 403.
            if not given.isascii():
                return None
            if access.owner and secrets.compare_digest(given, access.owner):
                return "owner"
            if access.guest and secrets.compare_digest(given, access.guest):
                return "guest"
            return None

        def do_GET(self):
            route = self._route()
            # Unauthenticated on purpose: the platform's health probe has no key,
            # and it reveals only that the process is up.
            if route == "/healthz":
                self._send(
                    200,
                    json.dumps({"ok": True, "seats": sorted(state.seats), "pushes": state.pushes}),
                    "application/json",
                )
                return
            role = self._role()
            if role is None:
                self._refuse(403, "a valid access key is required")
                return
            if route == "/state":
                query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
                seat = (query.get("seat") or [None])[0]
                payload = state.get(seat)
                payload["can_undo"] = False  # the board lives on the console, not here
                self._send(200, json.dumps(payload), "application/json")
            elif route == "/undo":
                self._refuse(405, "undo runs on the draft console, not the relay")
            else:
                self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            if self._route() != "/push":
                self._refuse(404, "not found")
                return
            if self._role() != "owner":
                self._refuse(403, "push is owner-only")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._refuse(400, "bad Content-Length")
                return
            if length <= 0 or length > MAX_PUSH_BYTES:
                self._refuse(413, "push body missing or too large")
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                seats = body["seats"]
                if not isinstance(seats, dict):
                    raise TypeError("seats must be an object")
            except (ValueError, KeyError, TypeError) as e:
                self._refuse(400, f"bad push: {e}")
                return
            n = state.push(seats)
            self._send(200, json.dumps({"ok": True, "seats": n}), "application/json")

    return Handler


def serve(
    state: RelayState,
    access: Access,
    port: int,
    host: str = "0.0.0.0",  # noqa: S104
) -> ThreadingHTTPServer:
    """Bind and start. Public by construction - the tokens are the guard."""
    server = ThreadingHTTPServer((host, port), make_handler(state, access))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> int:
    """Container entry point. Keys come from the environment, never a literal."""
    owner = os.environ.get("PUCKPILOT_OWNER_KEY", "")
    guest = os.environ.get("PUCKPILOT_GUEST_KEY", "")
    if not owner or not guest:
        print("PUCKPILOT_OWNER_KEY and PUCKPILOT_GUEST_KEY must both be set", flush=True)
        return 2
    port = int(os.environ.get("PORT", "8080"))
    serve(RelayState(), Access(owner=owner, guest=guest), port)
    print(f"relay listening on :{port}", flush=True)
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
