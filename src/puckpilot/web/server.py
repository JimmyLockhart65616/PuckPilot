"""Second-screen web view for draft night.

Deliberately stdlib-only (`http.server`): a draft-night tool should not need a
dependency install, a container, or a build step to come up.

The engine does not pick here. It shows a shortlist with the reasoning on both
sides, the full remaining board, the roster, and what the roster still needs —
and a human decides. Picks arrive on their own from the websocket feed, which
measured 190/190 against a real draft, so there is no manual pick entry to keep
in sync.

The page is the proof as much as the product: it shows how many picks the feed
has seen, whether any pick numbers are missing, and how long since the last one,
so a stalled feed is visible rather than silently frozen.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from puckpilot.draft.advice import market_disagreement, recommend
from puckpilot.draft.board import DraftBoard
from puckpilot.draft.engine import RosterValuePolicy
from puckpilot.draft.explain import summarize
from puckpilot.web.access import Access
from puckpilot.web.page import PAGE


def _fmt_cat(value: float, cat) -> str:
    """A counting stat as a whole number, a rate with enough decimals to differ."""
    return f"{value:.3f}".lstrip("0") if getattr(cat, "rate", False) else f"{value:.0f}"


@dataclass
class LiveState:
    """Board plus feed, guarded so HTTP threads and the poller can share it."""

    board: DraftBoard
    policy: RosterValuePolicy = field(default_factory=RosterValuePolicy)
    feed: object | None = None
    top: int = 3
    board_rows: int = 300
    # The league's scored categories, in its own order. Drives both the reason
    # text ("PPP", not "ppp") and the per-category line on each card, so a
    # different league shows its own categories without a code change.
    cats: tuple = ()

    @property
    def labels(self) -> dict[str, str]:
        return {c.key: c.label for c in self.cats}

    lock: threading.Lock = field(default_factory=threading.Lock)
    last_pick_at: float | None = None
    note: str = ""

    def pump(self) -> int:
        """Take whatever the feed has; returns how many picks landed."""
        if self.feed is None:
            return 0
        try:
            events = self.feed.poll(self.board)
        except Exception as e:
            self.note = f"feed error: {e.__class__.__name__}: {e}"
            return 0
        if not events:
            return 0
        from puckpilot.draft.feed import apply

        with self.lock:
            accepted, _rejected = apply(self.board, events)
        if accepted:
            self.last_pick_at = time.time()
        return len(accepted)

    def undo(self) -> str:
        """Recovery hatch: the feed is the only pick source, so a bad frame
        needs a way back without restarting the console mid-draft."""
        with self.lock:
            pick = self.board.undo()
        return f"undid {pick.name}" if pick else "nothing to undo"

    def snapshot(self, seat: int | None = None) -> dict:
        """The whole view for one seat.

        Two managers in the same room share one board - the picks are universal -
        but everything advisory here is answered against `seat`: the shortlist is
        roster-aware, and so is the board ordering, because `recommend` ranks on
        what the engine would actually take for *that* roster. So this is
        computed per seat rather than split into a shared half; the board rows
        are the bulk of the payload and they are not in fact shared.
        """
        seat = self.board.my_seat if seat is None else seat
        with self.lock:
            # Two questions, two lists. The shortlist answers "what do I take
            # now", so it obeys the roster rules. The board answers "what is
            # left", and must not: masking it hid every capped and every
            # non-needed position regardless of value, which is the one panel
            # whose whole job is to show the room's remaining supply.
            cands = recommend(self.board, self.policy, n=max(self.top * 8, 40), seat=seat)
            full = recommend(
                self.board,
                self.policy,
                n=self.board_rows,
                seat=seat,
                enforce_eligibility=False,
            )
            blocked = self.board.blocked(seat)
            labels = self.labels
            shortlist = [
                {
                    "name": c.name,
                    "position": c.position,
                    "team": c.team,
                    "vorp": c.vorp,
                    "adp_rank": c.adp_rank,
                    "p_survive": c.p_survive,
                    "reasons": [{"kind": r.kind, "text": r.text} for r in reasons],
                    # What he actually gets you, per category. Already computed
                    # on every Candidate and previously thrown away - it is the
                    # most direct answer to "why is this one better than that
                    # one" when the two look similar on VORP.
                    "projected": [
                        # Rates need their decimals: SV% rounded to an integer
                        # reads "1" and tells the drafter nothing.
                        [c2.label, _fmt_cat(c.projected[c2.key], c2)]
                        for c2 in self.cats
                        if c2.key in c.projected
                    ],
                }
                for c, reasons in summarize(self.board, cands, labels, top=self.top, seat=seat)
            ]
            board_rows = [
                {
                    "name": c.name,
                    "position": c.position,
                    "team": c.team,
                    "vorp": c.vorp,
                    "adp_rank": c.adp_rank,
                    "p_survive": c.p_survive,
                    # Why the rules would refuse him right now, if they would.
                    # Shown rather than hidden - see DraftBoard.blocked.
                    "blocked": blocked.get(c.position, ""),
                }
                for c in full
            ]
            on_clock = self.board.on_the_clock()
            nxt = self.board.next_pick_no(seat)
            roster = [{"name": p.name, "position": p.position} for p in self.board.roster(seat)]
            needs = [f"{p}x{n}" for p, n in sorted(self.board.needs(seat).items())]
            made, total = self.board.made, len(self.board.slots)
            rnd = self.board.current_round()
            # The real number of players still available, not the 300 the table
            # renders - it read "300 LEFT" for most of a draft.
            n_left = int(self.board.avail.sum())
            # Per-position supply, with the positions we still owe starters to
            # marked. The engine does not weight this (see
            # RosterValuePolicy._dynamic_vorp - measured, and left off), so it
            # is surfaced as a fact for the drafter to apply.
            sleeping, rated = market_disagreement(self.board, n=5, seat=seat)
            need_pos = set(self.board.needs(seat))
            supply = [[pos, n, pos in need_pos] for pos, n in sorted(self.board.supply().items())]

        status = self.feed.status() if hasattr(self.feed, "status") else {}
        return {
            "seat": seat,
            "round": rnd,
            "made": made,
            "total": total,
            "on_clock": on_clock,
            "my_turn": on_clock == seat,
            "picks_away": (None if nxt is None else nxt - made),
            "detected": status.get("picks_detected", made),
            "gaps": status.get("gaps", []),
            "n_left": n_left,
            "supply": supply,
            "market_gaps": {
                "sleeping": [asdict(g) for g in sleeping],
                "rated": [asdict(g) for g in rated],
            },
            "seconds_since_pick": (
                None if self.last_pick_at is None else time.time() - self.last_pick_at
            ),
            "shortlist": shortlist,
            "board": board_rows,
            "roster": roster,
            "needs": needs,
            "diagnostics": (
                f"feed={status.get('chosen', 'none')} frames={status.get('frames', 0)} "
                f"unmapped={status.get('unmapped', 0)}" + (f"  |  {self.note}" if self.note else "")
            ),
        }


def make_handler(state: LiveState, access: Access | None = None):
    access = access or Access()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console clean
            pass

        def _send(self, code, body, ctype):
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _route(self) -> str:
            """Path without its query string.

            Matched exactly, never by prefix: `startswith("/undo")` also accepts
            `/undo.png`, which is enough to smuggle a state change through an
            `<img>` tag.
            """
            return self.path.split("?", 1)[0].rstrip("/") or "/"

        def _token(self) -> str:
            """Access token from the query string or a header.

            The query string is what a shared link can actually carry; the header
            is for curl and for the console's own pushes.
            """
            header = self.headers.get("X-PuckPilot-Key")
            if header:
                return header
            q = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            return (q.get("k") or [""])[0]

        def _role(self) -> str | None:
            """ "owner", "guest", or None when the token does not match."""
            if not access.shared:
                return "owner"  # loopback-only: _is_same_origin is the guard
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

        def _seat(self) -> tuple[int | None, str | None]:
            """(seat, error). Absent means the board's own seat."""
            q = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            raw = (q.get("seat") or [None])[0]
            if raw is None:
                return state.board.my_seat, None
            try:
                seat = int(raw)
            except (TypeError, ValueError):
                return None, f"seat must be an integer, got {raw!r}"
            n = state.board.n_teams
            if not 0 <= seat < n:
                return None, f"seat {seat} outside 0..{n - 1}"
            return seat, None

        def _refuse(self, code: int, msg: str) -> None:
            self._send(code, json.dumps({"error": msg}), "application/json")

        def _is_same_origin(self) -> bool:
            """Reject cross-site requests to the mutating route.

            A GET or form POST from any other page the user has open is a
            "simple request": the browser sends it and the side effect fires
            even though the same-origin policy hides the reply. The port is a
            fixed default, so the target is guessable. Fetch metadata is the
            primary check; Origin is the fallback for clients that omit it.
            """
            # Host first. Sec-Fetch-Site alone does not survive DNS rebinding:
            # the attacker's own name resolves to 127.0.0.1, so the browser
            # truthfully reports same-origin while the page is theirs. Pinning
            # Host to the loopback names we actually serve closes that, and
            # also protects the read routes.
            # Read the port off the listening socket rather than trusting the
            # value passed in: serve(port=0) lets the OS choose, and comparing
            # against the literal 0 rejected every legitimate request.
            if access.shared:
                # The hostname is public now and a proxy rewrites Host, so the
                # pin cannot mean anything. The token is the guard instead; the
                # cross-site check below still runs.
                return self.headers.get("Sec-Fetch-Site") != "cross-site"
            bound = self.server.server_address[1]
            host = (self.headers.get("Host") or "").lower()
            if host and host not in (
                f"127.0.0.1:{bound}",
                f"localhost:{bound}",
                f"[::1]:{bound}",
            ):
                return False
            site = self.headers.get("Sec-Fetch-Site")
            if site is not None:
                return site in ("same-origin", "none")
            origin = self.headers.get("Origin")
            if origin is None:
                return True  # curl and friends: no browser, no CSRF vector
            return origin in (f"http://127.0.0.1:{bound}", f"http://localhost:{bound}")

        def do_GET(self):
            route = self._route()
            role = self._role()
            if route == "/state":
                # Guarded as well: /state is the entire board, roster and needs,
                # which is exactly what a rebinding attack would want to read.
                if not self._is_same_origin():
                    self._refuse(403, "cross-origin refused")
                    return
                if role is None:
                    self._refuse(403, "a valid access key is required")
                    return
                seat, err = self._seat()
                if err:
                    self._refuse(400, err)
                    return
                payload = state.snapshot(seat)
                # The page hides its undo button on this; the POST is gated
                # independently, so a forged value buys nothing.
                payload["can_undo"] = role == "owner"
                self._send(200, json.dumps(payload), "application/json")
            elif route == "/undo":
                # Mutating routes are POST-only, so a bare navigation or an
                # <img> src cannot rewind the board mid-draft.
                self._refuse(405, "use POST /undo")
            else:
                if role is None:
                    self._refuse(403, "a valid access key is required")
                    return
                self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            if self._route() != "/undo":
                self._refuse(404, "not found")
                return
            if not self._is_same_origin():
                self._refuse(403, "cross-origin refused")
                return
            # Owner only. A guest rewinding the shared board mid-draft is the
            # one destructive thing this view can do.
            if self._role() != "owner":
                self._refuse(403, "undo is owner-only")
                return
            self._send(200, json.dumps({"result": state.undo()}), "application/json")

    return Handler


def serve(
    state: LiveState,
    port: int = 8765,
    access: Access | None = None,
    host: str | None = None,
) -> ThreadingHTTPServer:
    """Start the server on a background thread; returns it so callers can stop it.

    Loopback-only by default: the board and roster are private and there is no
    auth, so nothing off this machine may reach it. Passing `access` is what
    opens it up, and it binds every interface only in that case - the tokens,
    not the bind address, are then the guard.
    """
    if host is None:
        host = "0.0.0.0" if (access and access.shared) else "127.0.0.1"  # noqa: S104
    server = ThreadingHTTPServer((host, port), make_handler(state, access))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
