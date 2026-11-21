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

**A decision can be lost, never repeated.** A decision lands here and waits
for the local job to drain it. Losing one costs a tap - the proposal reappears
on the next push for you to decide again - while inventing or replaying one
spends an acquisition, so every failure has to fall the first way, and the
local side treats a drained decision as the only decision. A drop tier set
from the roster (`/tier`) goes back the same way: lost, it is set again; it is
never made up.

Held in memory alone, a restart loses everything waiting. The hosted relay
scales to zero between runs, so there it keeps its state in one blob
(`PUCKPILOT_STATE_URL`) and keeps the rule: every write is conditional on the
blob still being what this process last read, and a drain stores the removal
before it hands anything over. A drain that cannot be stored hands over
nothing, and two processes overlapping during a deploy cannot both deliver.

No engine imports; this file is copied into a standard-library-only image.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from puckpilot.web import wire
from puckpilot.web.season_page import PAGE

MAX_PUSH_BYTES = 4 * 1024 * 1024

# A lineup that was right this morning is wrong by evening, so the page says
# how old it is and goes loud past this. Generous compared with the draft's 30
# seconds because the in-season job runs on a schedule, not continuously.
STALE_AFTER_S = 3 * 3600.0

# When a push says when the next run is due, the page is stale only once that
# moment has passed by this much - a run takes a minute or two, and a quiet
# night with nothing left to run is not a dead one.
NEXT_RUN_GRACE_S = 45 * 60.0

# More than this waiting to be drained means nobody is collecting them, which
# is worth failing loudly about rather than growing without bound.
MAX_PENDING_DECISIONS = 200

# The drop tiers a manager can set from the roster, and "auto" to go back to
# the rule. Mirrors `season.tiers` (this file imports no engine); a contract
# test holds the two together.
TIER_LABELS = {"never": "Never", "upgrade": "Upgrade only", "rotation": "Rotation"}
AUTO_TIER = "auto"

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


_SERVED: str | None = None


def served_build() -> str:
    """This process's `build_id`, worked out once - it is sent with every
    /state, so an open page can tell when the relay under it was redeployed."""
    global _SERVED
    if _SERVED is None:
        _SERVED = build_id()
    return _SERVED


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


def _stale(snap: dict, age: float, stale_after: float) -> bool:
    """Overdue, not merely old: judged against the next run the push promised."""
    due = snap.get("next_run_utc")
    if due:
        from datetime import datetime

        try:
            when = datetime.fromisoformat(str(due).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return age > stale_after
        return time.time() > when + NEXT_RUN_GRACE_S
    return age > stale_after


class StoreConflict(Exception):
    """The stored state changed after this process read it."""


class StoreUnavailable(RuntimeError):
    """The state blob could not be read or written. A RuntimeError, so the
    handler answers 503 and the local job simply tries again on its next run."""


class BlobStore:
    """The relay's state as one JSON blob, reached through a SAS URL.

    Every write is conditional - If-Match on the ETag last read, If-None-Match
    for the first - so one process can never overwrite what another wrote.
    """

    API_VERSION = "2021-08-06"

    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self.timeout = timeout

    def load(self) -> tuple[dict, str | None]:
        req = Request(self.url, headers={"x-ms-version": self.API_VERSION})
        try:
            with urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8")), r.headers.get("ETag")
        except HTTPError as e:
            if e.code == 404:
                return {}, None
            raise StoreUnavailable(f"state read failed: HTTP {e.code}") from e
        except (URLError, OSError, ValueError) as e:
            raise StoreUnavailable(f"state read failed: {e}") from e

    def save(self, data: dict, etag: str | None) -> str | None:
        headers = {
            "x-ms-version": self.API_VERSION,
            "x-ms-blob-type": "BlockBlob",
            "Content-Type": "application/json",
        }
        if etag:
            headers["If-Match"] = etag
        else:
            headers["If-None-Match"] = "*"
        body = wire.dumps(data).encode("utf-8")
        req = Request(self.url, data=body, method="PUT", headers=headers)
        try:
            with urlopen(req, timeout=self.timeout) as r:
                return r.headers.get("ETag")
        except HTTPError as e:
            if e.code in (409, 412):
                raise StoreConflict from e
            raise StoreUnavailable(f"state write failed: HTTP {e.code}") from e
        except (URLError, OSError) as e:
            raise StoreUnavailable(f"state write failed: {e}") from e


class SeasonState:
    """The last snapshot per manager, and the decisions waiting to be collected.

    Without a store it all lives in memory. With one, every call starts from
    what is stored, and every change is stored before it is answered.
    """

    def __init__(self, stale_after: float = STALE_AFTER_S, store: BlobStore | None = None):
        self._lock = threading.Lock()
        self._snap: dict[str, dict] = {}
        self._at: dict[str, float] = {}
        self._decisions: list[dict] = []
        self._seq = 0
        self._store = store
        self._etag: str | None = None
        self.stale_after = stale_after
        self.started = time.time()

    # -- storage -----------------------------------------------------------

    def _read(self) -> None:
        """Replace what is held with what is stored. The caller holds the lock."""
        if self._store is None:
            return
        data, self._etag = self._store.load()
        self._snap = data.get("snap", {})
        self._at = data.get("at", {})
        self._decisions = data.get("decisions", [])
        self._seq = data.get("seq", 0)

    def _change(self, fn):
        """Apply `fn` to the current state and store the result before returning.

        A write that loses a race starts over from what the winner stored, so
        nothing is applied twice and nothing a drain removed comes back. A write
        that fails outright leaves the change unstored - and, because every call
        starts by reading, unapplied.
        """
        with self._lock:
            for _ in range(5):
                self._read()
                result = fn()
                if self._store is None:
                    return result
                stored = {
                    "v": 1,
                    "snap": self._snap,
                    "at": self._at,
                    "decisions": self._decisions,
                    "seq": self._seq,
                }
                try:
                    self._etag = self._store.save(stored, self._etag)
                except StoreConflict:
                    continue
                return result
            raise StoreUnavailable("the stored state kept changing underneath this write")

    # -- the view ----------------------------------------------------------

    def push(self, manager: str, snapshot: dict) -> None:
        def keep():
            self._snap[manager] = snapshot
            self._at[manager] = time.time()

        self._change(keep)

    def get(self, manager: str) -> dict:
        with self._lock:
            self._read()
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
                "alerts": [],
                "next_run_utc": None,
                "next_local": "",
                "acted": None,
                "withdrawn": [],
                "blockers": [],
                "ledger": [],
                "build": served_build(),
            }
        age = time.time() - at
        out = dict(snap)
        out["manager"] = manager
        out["empty"] = False
        out["age_seconds"] = round(age, 1)
        out["stale"] = _stale(snap, age, self.stale_after)
        out["build"] = served_build()
        return out

    # -- the return path ---------------------------------------------------

    def decide(
        self, manager: str, kind: str, ident: int, approve: bool, executes: bool = False
    ) -> dict:
        if kind not in ("proposal", "protocol"):
            raise ValueError(f"unknown decision kind {kind!r}")

        def record() -> dict:
            if len(self._decisions) >= MAX_PENDING_DECISIONS:
                raise RuntimeError("too many undelivered decisions; is the local job running?")
            self._seq += 1
            row = {
                "seq": self._seq,
                "manager": manager,
                "kind": kind,
                "id": int(ident),
                "approve": bool(approve),
                # What the page told the person Approve would do: make the move
                # (True) or leave it to them. Carried, never inferred.
                "executes": bool(executes),
                "at": time.time(),
            }
            self._decisions.append(row)
            # Reflect it immediately so the page does not show a decided item
            # as still pending until the next push.
            self._mark_decided(manager, kind, int(ident), bool(approve))
            return row

        return self._change(record)

    def set_tier(self, manager: str, player_key: str, tier: str) -> dict:
        """Queue a drop tier for one rostered player, and show it at once.

        Only a player on the roster this manager was last shown, and never one
        the platform locks (`tier_source` "yahoo") - no pin can move him.
        """
        if tier != AUTO_TIER and tier not in TIER_LABELS:
            known = ", ".join([*TIER_LABELS, AUTO_TIER])
            raise ValueError(f"unknown tier {tier!r} (known: {known})")

        def record() -> dict:
            snap = self._snap.get(manager) or {}
            row = next(
                (r for r in snap.get("roster", []) if r.get("player_key") == player_key), None
            )
            if row is None:
                raise ValueError("that player is not on the roster shown")
            if row.get("tier_source") == "yahoo":
                raise ValueError(f"Yahoo does not let {row.get('name', 'him')} be dropped")
            if len(self._decisions) >= MAX_PENDING_DECISIONS:
                raise RuntimeError("too many undelivered decisions; is the local job running?")
            self._seq += 1
            out = {
                "seq": self._seq,
                "manager": manager,
                "kind": "tier",
                "player": player_key,
                "tier": tier,
                "at": time.time(),
            }
            self._decisions.append(out)
            # Shown as set until the next push, which says what the run made of it.
            if tier == AUTO_TIER:
                row.update(tier_source="auto", tier_why="automatic - from PuckPilot's next run")
            else:
                row.update(
                    tier=tier,
                    tier_label=TIER_LABELS[tier],
                    tier_source="you",
                    tier_why="set by you - used from PuckPilot's next run",
                )
            return out

        return self._change(record)

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
        """Hand over this manager's decisions exactly once.

        They are stored as gone before they are handed over, so a failure on
        the way loses them rather than repeating them.
        """
        with self._lock:
            # Most runs find nothing waiting; they need not write to find that out.
            self._read()
            if not any(d["manager"] == manager for d in self._decisions):
                return []

        def take() -> list[dict]:
            mine = [d for d in self._decisions if d["manager"] == manager]
            self._decisions = [d for d in self._decisions if d["manager"] != manager]
            return mine

        return self._change(take)

    def health(self) -> dict:
        with self._lock:
            # The process is up either way; a storage fault shows on /state, as a 503.
            with contextlib.suppress(StoreUnavailable):
                self._read()
            managers = sorted(self._snap)
            waiting = len(self._decisions)
        return {
            "ok": True,
            "build": build_id(),
            "git": os.environ.get("PUCKPILOT_GIT_SHA", ""),
            "uptime_s": round(time.time() - self.started, 1),
            "managers": len(managers),
            "decisions_waiting": waiting,
            "stored": self._store is not None,
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
            if route not in ("/state", "/decisions"):
                self._send(200, PAGE, "text/html; charset=utf-8")
                return
            try:
                if route == "/state":
                    out = state.get(manager)
                else:
                    out = {"decisions": state.drain(manager)}
            except RuntimeError as e:
                self._refuse(503, str(e))
                return
            self._send(200, wire.dumps(out), "application/json")

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
                try:
                    state.push(manager, wire.clean(snap))
                except RuntimeError as e:
                    self._refuse(503, str(e))
                    return
                self._send(200, wire.dumps({"ok": True, "manager": manager}), "application/json")
            elif route == "/tier":
                try:
                    row = state.set_tier(
                        manager, str(body.get("player") or ""), str(body.get("tier") or "")
                    )
                except (TypeError, ValueError) as e:
                    self._refuse(400, str(e))
                except RuntimeError as e:
                    self._refuse(503, str(e))
                else:
                    self._send(200, wire.dumps({"ok": True, "seq": row["seq"]}), "application/json")
            elif route == "/decide":
                try:
                    row = state.decide(
                        manager,
                        str(body.get("kind")),
                        int(body.get("id")),
                        bool(body.get("approve")),
                        executes=body.get("executes") is True,
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
    # A SAS URL for one blob, from a container app secret. Unset, state lives in
    # memory and is gone on restart - fine locally, not when scaled to zero.
    url = os.environ.get("PUCKPILOT_STATE_URL", "")
    store = BlobStore(url) if url else None
    serve(SeasonState(store=store), keys, port)
    where = "blob storage" if store else "memory"
    print(
        f"season relay listening on :{port} for {len(keys)} manager(s), state in {where}",
        flush=True,
    )
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
