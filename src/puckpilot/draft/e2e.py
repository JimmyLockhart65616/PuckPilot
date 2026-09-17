"""A whole draft through every hop a second manager depends on.

`draft live --web --publish` has four hops between a pick and a guest's screen:
the board records it, the console builds a snapshot, the snapshot is pushed to
the relay, and the relay hands it to a browser. Each hop has been tested alone.
This drives complete drafts through all of them at once and checks, at every
pick, that each hop agrees with the one before it:

1. the snapshot agrees with the board (`integrity.snapshot_violations`);
2. the console's own `/state` over HTTP agrees with the snapshot;
3. the push succeeds, through the console's real `_push_snapshots`;
4. the relay's `/state`, read with the guest key, agrees with what was pushed;
5. for picks that came from a real room, the player the board struck off is
   the player Yahoo said was drafted.

A relay can be started in-process or named by URL, so the same run rehearses
the Azure deployment. Picks come from bots on the real league, from a harvested
Yahoo room, or from the raw websocket frames of a recorded draft pushed through
the real frame parser - never from a live room.

Two kinds of failure are kept apart. A *violation* is the view disagreeing with
itself - a wrong screen - and fails the run. *Drift* is the board disagreeing
with the room (a pick we could not map), which is real and is reported, but is
a data-coverage question rather than an integrity one.
"""

from __future__ import annotations

import difflib
import json
import os
import secrets
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from puckpilot.draft import integrity
from puckpilot.draft.board import DraftBoard
from puckpilot.draft.feed import PickEvent, apply
from puckpilot.keepers import _norm
from puckpilot.web import wire
from puckpilot.web.server import LiveState

Progress = Callable[[str], None]

# Two spellings of one player ("Mitch"/"Mitchell Marner", MoneyPuck's accent
# loss "Sttzle") stay above this; two different players do not.
SAME_PLAYER_RATIO = 0.75
HTTP_TIMEOUT_S = 15.0


# ---- relay endpoints ----------------------------------------------------------


@dataclass
class Relay:
    url: str
    owner: str
    guest: str
    local: object | None = None  # the in-process server, when we started one

    def close(self) -> None:
        if self.local is not None:
            self.local.shutdown()
            self.local.server_close()


def start_local_relay() -> Relay:
    from puckpilot.web.access import Access
    from puckpilot.web.relay import RelayState
    from puckpilot.web.relay import serve as serve_relay

    owner, guest = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
    srv = serve_relay(RelayState(), Access(owner=owner, guest=guest), port=0, host="127.0.0.1")
    return Relay(f"http://127.0.0.1:{srv.server_address[1]}", owner, guest, local=srv)


def _http(url: str, method: str = "GET", body: bytes | None = None, headers=None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


# ---- result -------------------------------------------------------------------


@dataclass
class E2EResult:
    name: str
    source: str
    picks: int = 0
    total: int = 0
    room_picks: int | None = None
    checks: int = 0
    pushes: int = 0
    browser_checks: int = 0
    violations: list[str] = field(default_factory=list)
    drift: list[str] = field(default_factory=list)
    unmapped: int = 0
    unknown: int = 0
    names_checked: int = 0
    # Drafted players whose board position Yahoo does not allow. Not a failure:
    # the right player, positioned by a different source. Surfaced because it
    # decides whether "starts right away at C" is true.
    eligibility: set[str] = field(default_factory=set)
    timings: dict[str, list[float]] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.violations

    def time(self, hop: str, ms: float) -> None:
        self.timings.setdefault(hop, []).append(ms)

    def summary(self) -> str:
        hops = "  ".join(
            f"{k} {statistics.median(v):.0f}/{max(v):.0f}ms"
            for k, v in sorted(self.timings.items())
        )
        room = "" if self.room_picks is None else f" of {self.room_picks} in the room"
        verdict = "PASS" if self.passed else f"FAIL ({len(self.violations)} violations)"
        lines = [
            f"{verdict}  {self.name} [{self.source}]  {self.picks}/{self.total} picks{room}, "
            f"{self.checks} checks, {self.pushes} pushes, {self.browser_checks} browser checks, "
            f"{self.names_checked} names matched, "
            f"{self.unknown} unrankable, {self.unmapped} unmapped, {self.seconds:.0f}s",
            f"      hops (median/max): {hops}",
        ]
        lines += [f"      DRIFT: {d}" for d in self.drift]
        if self.eligibility:
            lines.append(
                f"      ELIGIBILITY ({len(self.eligibility)}): "
                + "; ".join(sorted(self.eligibility)[:6])
                + (" ..." if len(self.eligibility) > 6 else "")
            )
        lines += [f"      VIOLATION: {v}" for v in self.violations[:15]]
        if len(self.violations) > 15:
            lines.append(f"      ... {len(self.violations) - 15} more")
        return "\n".join(lines)


# ---- pick sources ---------------------------------------------------------------


class _NoContext:
    """Stands in for a Playwright context: frames arrive through `ingest`."""

    pages: list = []

    def on(self, *_a, **_k) -> None:
        pass


def capture_frames(capture_dir: Path) -> list[str]:
    """The frames the room broadcast, in order, from a `draft capture` recording."""
    frames = []
    path = Path(capture_dir) / "websocket.jsonl"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("dir") == "recv" and isinstance(row.get("payload"), str):
            frames.append(row["payload"])
    return frames


def frame_polls(frames: list[str], feed) -> Iterator[Callable[[DraftBoard], list[PickEvent]]]:
    """Push frames into a real WebsocketFeed one at a time, as the socket would."""
    for payload in frames:
        feed.ingest(payload)
        yield feed.poll


@dataclass(frozen=True)
class YahooRow:
    name: str
    positions: frozenset[str]


# Yahoo's position codes, in the board's single-letter vocabulary.
_YAHOO_POS = {"C": "C", "LW": "L", "RW": "R", "D": "D", "G": "G"}


def yahoo_rows(conn) -> dict[int, YahooRow]:
    """NHL id -> the Yahoo row that maps to him.

    Keyed the way the feed resolves a pick, so checking a recorded pick against
    this checks the MAPPING: if Yahoo's player were resolved to the wrong NHL id,
    the name or the eligibility here would disagree with what the board struck.
    """
    out = {}
    for nid, name, positions in conn.execute(
        "SELECT nhl_player_id, full_name, positions FROM yahoo_player_map"
        " WHERE nhl_player_id IS NOT NULL"
    ):
        codes = {_YAHOO_POS[p] for p in str(positions or "").split(",") if p in _YAHOO_POS}
        out[int(nid)] = YahooRow(str(name), frozenset(codes))
    return out


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _surname(name: str) -> str:
    parts = [_norm(p) for p in name.split()]
    parts = [p for p in parts if p and p not in _SUFFIXES]
    return parts[-1] if parts else ""


def same_player(a: str, b: str) -> bool:
    """Two spellings of one player, not two players.

    The surname has to agree on its own: over the whole name a shared first
    name carries the ratio ("Sebastian Aho" vs "Sebastian Cossa" is 0.77).
    """
    x, y = _norm(a), _norm(b)
    if x == y:
        return True
    sa, sb = _surname(a), _surname(b)
    if sa and sa == sb:
        # Same surname, and a given name that is a short form of the other:
        # "Nick"/"Nicholas" Paul scores 0.70 overall, below any safe ratio.
        fa, fb = x.removesuffix(sa), y.removesuffix(sb)
        common = len(os.path.commonprefix([fa, fb]))
        if common >= 3 or difflib.SequenceMatcher(None, x, y).ratio() >= SAME_PLAYER_RATIO:
            return True
    ratio = difflib.SequenceMatcher(None, x, y).ratio()
    surname = difflib.SequenceMatcher(None, sa, sb).ratio()
    return ratio >= SAME_PLAYER_RATIO and surname >= SAME_PLAYER_RATIO


# ---- the guest's actual browser ---------------------------------------------------


class BrowserProbe:
    """A real headless browser on the guest link, polling like the second manager.

    The Node test runs the page's script against payloads; this runs the page
    itself, over the real network path, with the browser's own fetch, JSON
    parser and DOM. A fresh temporary profile every time - never the logged-in
    Yahoo profile, which a live console may be using.
    """

    def __init__(self, url: str, channel: str = "chrome"):
        from playwright.sync_api import sync_playwright

        self.url = url
        self.errors: list[str] = []
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True, channel=channel)
        self.page = self._browser.new_page()
        self.page.on("pageerror", lambda e: self.errors.append(f"page error: {e}"))
        self.page.on("console", self._console)
        self.page.on("response", self._response)
        self.page.goto(url, wait_until="domcontentloaded")

    def _console(self, msg) -> None:
        # A failed request is logged twice by the browser - once here without
        # its URL - so it is recorded from the response instead.
        if msg.type == "error" and not msg.text.startswith("Failed to load resource"):
            self.errors.append(f"console: {msg.text}")

    def _response(self, response) -> None:
        from urllib.parse import urlparse

        # Path only: the query string carries the guest key.
        path = urlparse(response.url).path
        # The browser asks for a favicon on its own, without the key the page
        # carries, and the relay rightly refuses it.
        if response.status >= 400 and path != "/favicon.ico":
            self.errors.append(f"HTTP {response.status} on {path}")

    def shows_pick(self, made: int, total: int, timeout_s: float = 12.0) -> str:
        """ "" if the page caught up to `made`, else what it shows instead."""
        want = f"{made}/{total}" if total and made >= total else f"{made + 1}/{total}"
        try:
            self.page.wait_for_function(
                "w => document.getElementById('pick').textContent === w",
                arg=want,
                timeout=timeout_s * 1000,
            )
        except Exception:
            shown = self.page.text_content("#pick")
            banner = (self.page.text_content("#banner") or "").strip()
            return f"browser shows pick {shown!r}, expected {want!r} (banner: {banner[:100]!r})"
        text = self.page.inner_text("body")
        for tell in ("undefined", "NaN"):
            if tell in text:
                return f"browser page contains {tell!r}"
        if not self.page.is_hidden("#banner"):
            return f"browser shows a banner: {self.page.text_content('#banner')!r}"
        return ""

    def says_not_live(self, timeout_s: float) -> bool:
        try:
            self.page.wait_for_function(
                "() => document.getElementById('banner').textContent.includes('NOT LIVE')",
                timeout=timeout_s * 1000,
            )
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()


# ---- the run ----------------------------------------------------------------------


class Harness:
    """One draft, one console, one relay."""

    def __init__(
        self,
        board: DraftBoard,
        relay: Relay,
        cats: tuple = (),
        seats: tuple[int, ...] = (0,),
        every: int = 1,
        top: int = 3,
        board_rows: int = 300,
        names: dict[int, YahooRow] | None = None,
        expect_build: str | None = None,
        browser_every: int = 0,
        stale_check: bool = False,
        progress: Progress = print,
    ):
        self.board = board
        self.relay = relay
        self.seats = tuple(seats)
        self.every = max(1, every)
        self.names = names or {}
        self.expect_build = expect_build
        # 0 = no browser. Otherwise a real headless browser on the guest link is
        # checked every N picks and at the end.
        self.browser_every = browser_every
        self.stale_check = stale_check
        self.progress = progress
        self.state = LiveState(board=board, feed=None, top=top, board_rows=board_rows, cats=cats)

    # -- one verification pass -------------------------------------------------

    def _verify(self, result: E2EResult, local_url: str) -> None:
        from puckpilot.cli import _push_snapshots

        board, state = self.board, self.state
        before: dict[int, dict] = {}
        for seat in self.seats:
            t = time.perf_counter()
            snap = state.snapshot(seat)
            result.time("snapshot", (time.perf_counter() - t) * 1000)
            before[seat] = snap
            result.violations += integrity.snapshot_violations(
                board, snap, seat, state.policy, top=state.top, board_rows=state.board_rows
            )
            t = time.perf_counter()
            status, text = _http(f"{local_url}/state?seat={seat}")
            result.time("local_http", (time.perf_counter() - t) * 1000)
            if status != 200:
                result.violations.append(f"local /state seat {seat}: HTTP {status} {text[:120]}")
                continue
            try:
                served = wire.loads(text)
            except ValueError as e:
                result.violations.append(f"local /state seat {seat}: not strict JSON ({e})")
                continue
            result.violations += integrity.relay_violations(
                snap, served, where=f"local http seat {seat} pick {board.made}"
            )

        t = time.perf_counter()
        err = _push_snapshots(self.relay.url, self.relay.owner, state, list(self.seats))
        result.time("push", (time.perf_counter() - t) * 1000)
        result.pushes += 1
        if err:
            result.violations.append(f"push at pick {board.made} failed: {err}")
            return

        for seat in self.seats:
            t = time.perf_counter()
            status, text = _http(f"{self.relay.url}/state?seat={seat}&k={self.relay.guest}")
            result.time("relay_read", (time.perf_counter() - t) * 1000)
            where = f"relay seat {seat} pick {board.made}"
            if status != 200:
                result.violations.append(f"{where}: HTTP {status} {text[:120]}")
                continue
            try:
                remote = wire.loads(text)
            except ValueError as e:
                result.violations.append(f"{where}: not strict JSON ({e})")
                continue
            result.violations += integrity.relay_violations(before[seat], remote, where=where)
            if remote.get("stale"):
                result.violations.append(f"{where}: relay reports stale right after a push")
            if remote.get("can_undo") is not False:
                result.violations.append(f"{where}: relay offers undo to a guest")
        result.checks += 1

    def _boundary(self, result: E2EResult) -> None:
        """The relay's guard rails, checked against the real endpoint once per run."""
        url = self.relay.url
        status, text = _http(f"{url}/healthz")
        if status != 200:
            result.violations.append(f"relay /healthz: HTTP {status}")
        else:
            health = json.loads(text)
            if self.expect_build and health.get("build") != self.expect_build:
                result.violations.append(
                    f"relay runs build {health.get('build')!r}, this checkout is "
                    f"{self.expect_build!r} - the deployment is not this code"
                )
        if _http(f"{url}/state")[0] != 403:
            result.violations.append("relay /state answers without a key")
        if _http(f"{url}/state?k=not-the-key")[0] != 403:
            result.violations.append("relay /state answers a wrong key")
        guest_push = _http(
            f"{url}/push",
            "POST",
            b'{"seats": {"0": {"made": 999}}}',
            {"X-PuckPilot-Key": self.relay.guest, "Content-Type": "application/json"},
        )
        if guest_push[0] != 403:
            result.violations.append(f"relay let a guest push (HTTP {guest_push[0]})")
        if _http(f"{url}/undo?k={self.relay.owner}", "POST", b"")[0] == 200:
            result.violations.append("relay accepted an undo")

    def _look(self, result: E2EResult, probe: BrowserProbe) -> None:
        t = time.perf_counter()
        problem = probe.shows_pick(self.board.made, len(self.board.slots))
        result.time("browser", (time.perf_counter() - t) * 1000)
        result.browser_checks += 1
        if problem:
            result.violations.append(f"guest browser at pick {self.board.made}: {problem}")

    # -- the draft ----------------------------------------------------------------

    def run(
        self,
        polls: Iterator,
        name: str,
        source: str,
        room_picks: int | None = None,
        unmapped: Callable[[], list[str]] | None = None,
    ) -> E2EResult:
        from puckpilot.web.server import serve

        result = E2EResult(name=name, source=source, room_picks=room_picks)
        board = self.board
        result.total = len(board.slots)
        t0 = time.perf_counter()
        local = serve(self.state, port=0)
        local_url = f"http://127.0.0.1:{local.server_address[1]}"
        probe = None
        try:
            self._boundary(result)
            self._verify(result, local_url)
            if self.browser_every:
                probe = BrowserProbe(f"{self.relay.url}/?seat={self.seats[0]}&k={self.relay.guest}")
                self._look(result, probe)
            for poll in polls:
                if board.complete:
                    break
                events = poll(board) if callable(poll) else poll
                if not events:
                    continue
                accepted, rejected = apply(board, events)
                result.unknown += sum(1 for r in rejected if "slot consumed" in r)
                for msg in rejected:
                    if "slot consumed" not in msg:
                        result.violations.append(f"pick {board.made}: feed pick refused: {msg}")
                if accepted or rejected:
                    self.state.last_pick_at = time.time()
                for pick in accepted:
                    row = self.names.get(int(pick.player_id))
                    if row is None:
                        continue
                    result.names_checked += 1
                    if not same_player(row.name, pick.name):
                        result.violations.append(
                            f"pick {pick.overall + 1}: the room drafted {row.name!r}, "
                            f"the board struck off {pick.name!r}"
                        )
                    elif row.positions and pick.position not in row.positions:
                        # Right player, and a known difference in vocabulary:
                        # MoneyPuck's one position against Yahoo's eligibility.
                        result.eligibility.add(
                            f"{pick.name}: board {pick.position}, "
                            f"Yahoo {'/'.join(sorted(row.positions))}"
                        )
                if board.made % self.every == 0 or board.complete:
                    self._verify(result, local_url)
                    if probe and board.made % self.browser_every == 0:
                        self._look(result, probe)
            if board.made % self.every:
                self._verify(result, local_url)
            if probe:
                self._look(result, probe)
                if self.stale_check:
                    # Stop pushing, as a console that died would. The guest
                    # must be told within a few heartbeats.
                    from puckpilot.web.relay import STALE_AFTER_S

                    if not probe.says_not_live(STALE_AFTER_S + 20):
                        result.violations.append(
                            "browser never said NOT LIVE after the console stopped pushing"
                        )
                    result.browser_checks += 1
                result.violations += probe.errors
        finally:
            if probe:
                probe.close()
            local.shutdown()
            local.server_close()

        result.picks = board.made
        missing = unmapped() if unmapped else []
        result.unmapped = len(missing)
        if room_picks is not None:
            expected = min(room_picks, len(board.slots))
            if board.made < expected:
                result.drift.append(
                    f"board ended {expected - board.made} pick(s) behind the room; "
                    f"unmapped Yahoo ids: {missing[:10]}"
                )
        elif not board.complete:
            result.violations.append(f"draft stopped at {board.made}/{len(board.slots)}")
        result.seconds = time.perf_counter() - t0
        return result


# ---- convenience drivers used by the CLI and the tests --------------------------


def sim_polls(board: DraftBoard, league, rng, our_policy=None) -> Iterator:
    """Bots in every seat, ours included - the field the draft sim is scored on."""
    from puckpilot.draft.engine import RosterValuePolicy
    from puckpilot.draft.feed import SimFeed
    from puckpilot.draft.sim import _default_opponents

    opponents = _default_opponents(rng, league)
    order = rng.permutation(len(opponents))
    bots, oi = [], 0
    for s in range(board.n_teams):
        if s == board.my_seat:
            bots.append(our_policy or RosterValuePolicy())
        else:
            bots.append(opponents[order[oi % len(opponents)]])
            oi += 1
    feed = SimFeed(bots, rng)
    guard = len(board.slots) * 3
    for _ in range(guard):
        if board.complete:
            return
        yield feed.poll


def replay_polls(feed, n: int) -> Iterator:
    for _ in range(n):
        yield feed.poll


def room_league(league, n_teams: int):
    """The league, reshaped to a harvested room: its team count, no keepers."""
    from dataclasses import replace

    return replace(
        league,
        shape=replace(league.shape, n_teams=n_teams),
        n_keepers=0,
        keepers_by_season={},
        keeper_owners_by_season={},
    )
