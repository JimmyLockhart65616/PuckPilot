"""Sit through Yahoo mock drafts and harvest what they teach.

Mock drafts are the only source of *real human draft behaviour* available before
the real draft. That is not evidence about whether we draft well - a mock drafts
2026-27 players and no 2026-27 season has been played, so there is no outcome to
score against until spring.

What a mock does measure, directly and without circularity, is **when players
actually go**. That calibrates `RosterValuePolicy.survival_spread`, currently the
guessed constant 6.0 fitted against a *simulated* ADP bot field - and STATUS.md
credits `survival_discount` with the engine's entire measured edge. Each mock
yields ~192 observations of the room's behaviour to replace that guess with data.

It also exposes pool coverage: players the room drafts that our board does not
contain at all (Gavin McKenna at ADP 92, Ivar Stenberg at 139 in the 2026-09-08
run). Those are invisible to any amount of offline simulation.

Conduct
-------
A public mock draft contains real people, so:

- **This never picks.** It reads the frames the room broadcasts and writes them
  to disk. The seat is the user's own, played (or autopicked) by Yahoo exactly
  as if the tool were not running - the recording changes nothing about how the
  draft goes for anyone in the room.
- **`MAX_RUNS` caps a session.** An unattended seat is worth less to the other
  eleven players than a live one, so this is metered rather than left to loop.
- **Mock rooms only.** `LOBBY` is the mock lobby by construction. Do not point
  this at a real league: on draft night the engine advises and a human decides.
- **Picks only.** The room also broadcasts Yahoo's own advice signals - its ADP,
  tiering and dropoff. Those are Yahoo's analytics product rather than an
  observation of the draft, and the calibration does not need them, so they are
  not recorded.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from puckpilot.draft.wsfeed import PickFrame, WebsocketFeed, parse_frame, pump

LOBBY = "https://hockey.fantasysports.yahoo.com/hockey/mock_lobby"
# A harvest session is capped rather than open-ended: each run occupies a seat in
# a room of real people. Ten mocks is ~1,900 survival observations, which is
# already past where the calibration fit stops moving.
MAX_RUNS = 10
# A mock room is idle far longer than a draft takes; give up rather than hang.
JOIN_TIMEOUT_S = 900.0
DRAFT_TIMEOUT_S = 3600.0

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


@dataclass
class MockResult:
    """One completed mock, reduced to what is worth keeping."""

    started: str
    picks: list[dict] = field(default_factory=list)  # pick, yahoo_id, seat, pos
    adp_observations: list[dict] = field(default_factory=list)
    n_teams: int = 0
    rounds: int = 0
    completed: bool = False
    note: str = ""

    @property
    def summary(self) -> str:
        return (
            f"{len(self.picks)} picks, {self.n_teams} teams, "
            f"{len(self.adp_observations)} ADP rows"
            f"{'' if self.completed else '  (INCOMPLETE: ' + self.note + ')'}"
        )


class MockRecorder:
    """Collects the pick frames from one draft."""

    def __init__(self):
        self.picks: dict[int, PickFrame] = {}
        self.started = datetime.now(UTC).isoformat()

    def ingest(self, payload: str) -> None:
        """Record picks. Nothing else on the socket is kept.

        An earlier version also lifted Yahoo's own advice signals - its ADP,
        tiering, VOLS rank and dropoff - off a separate channel the room
        broadcasts. That was Yahoo's analytics product rather than an
        observation of the draft, and it is not needed: the calibration fits
        against ADP from the player map, which covered 192 of 192 picks in the
        2026-09-08 mock where the in-draft channel reached 26. Taking it bought
        nothing measurable, so it is not taken.
        """
        frame = parse_frame(str(payload))
        if isinstance(frame, PickFrame):
            self.picks[frame.pick] = frame

    def result(self, n_teams: int, completed: bool, note: str = "") -> MockResult:
        picks = [asdict(self.picks[k]) | {"pick": k} for k in sorted(self.picks)]
        rounds = (max(self.picks) // n_teams) if (self.picks and n_teams) else 0
        return MockResult(
            started=self.started,
            picks=picks,
            # Retained on the dataclass so harvests banked before the advice
            # channel was dropped still load; nothing writes it now.
            adp_observations=[],
            n_teams=n_teams,
            rounds=rounds,
            completed=completed,
            note=note,
        )


def infer_teams(feed: WebsocketFeed) -> int:
    """Team count from the clock frames, rather than assuming twelve."""
    seats = {f.seat for f in feed.state.picks.values()}
    return max(seats) if seats else 0


def run_one(
    context,
    conn,
    league,
    yahoo_to_nhl: dict[str, int],
    seat_hint: int | None = None,
    progress: Progress = _noop,
    join_timeout: float = JOIN_TIMEOUT_S,
    draft_timeout: float = DRAFT_TIMEOUT_S,
) -> MockResult:
    """Sit through one mock draft, recording everything it emits.

    Returns whatever was captured even when the draft never starts — an
    abandoned lobby is a normal outcome and must not lose the runs before it.
    """
    recorder = MockRecorder()
    feed = WebsocketFeed(context, yahoo_to_nhl)
    original_ingest = feed.ingest

    def tee(payload: str) -> None:
        recorder.ingest(payload)
        original_ingest(payload)

    feed.ingest = tee  # type: ignore[method-assign]

    progress("  waiting for the draft to start...")
    start = time.time()
    while not feed.state.picks:
        if time.time() - start > join_timeout:
            return recorder.result(0, False, "draft never started")
        pump(context, 2.0)

    progress("  drafting...")
    last_seen, stalled_since = 0, time.time()
    while time.time() - start < draft_timeout:
        pump(context, 2.0)
        n = feed.state.highest_pick
        if n != last_seen:
            last_seen, stalled_since = n, time.time()
            if n % 24 == 0:
                progress(f"    pick {n}")
        # A finished draft simply stops emitting; there is no "done" frame.
        elif time.time() - stalled_since > 90:
            break

    teams = infer_teams(feed)
    complete = bool(feed.state.picks) and not feed.state.gaps
    note = "" if complete else f"gaps at {feed.state.gaps[:5]}"
    return recorder.result(teams, complete, note)


def harvest_capture(capture_dir: Path) -> MockResult:
    """Turn an existing `ppilot draft capture` recording into a harvest.

    A capture of a completed draft holds exactly the frames a live run would
    have seen, so it is worth the same to the calibration - and it costs nobody
    a seat in a mock room to replay one.

    Only `recv` frames are read. The `sent` side carries our own browser's
    identifiers (the handshake frame embeds the full user-agent string), and
    none of it is anything the room broadcast.
    """
    path = Path(capture_dir) / "websocket.jsonl"
    recorder = MockRecorder()
    seats: set[int] = set()
    if not path.is_file():
        return recorder.result(0, False, f"no websocket.jsonl in {capture_dir}")

    first_wall = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("dir") != "recv":
            continue
        payload = row.get("payload")
        if not isinstance(payload, str):
            continue
        if not first_wall and isinstance(row.get("wall"), str):
            first_wall = row["wall"]
        recorder.ingest(payload)

    # Stamp the harvest with when the DRAFT happened, not when it was replayed.
    # `save` names the file from this, so re-banking the same capture overwrites
    # rather than double-counting that draft in the calibration.
    if first_wall:
        recorder.started = first_wall

    for frame in recorder.picks.values():
        seats.add(frame.seat)
    teams = max(seats) if seats else 0
    gaps = [n for n in range(1, max(recorder.picks, default=0) + 1) if n not in recorder.picks]
    return recorder.result(
        teams, bool(recorder.picks) and not gaps, f"gaps at {gaps[:5]}" if gaps else ""
    )


def save(result: MockResult, root: Path) -> Path | None:
    """Write a harvest to disk. A draft that never started is not written.

    An abandoned lobby produces a zero-pick result. Keeping those would inflate
    the draft count the calibration report leans on when it says how much
    evidence a spread rests on.
    """
    if not result.picks:
        return None
    root.mkdir(parents=True, exist_ok=True)
    stamp = result.started.replace(":", "").replace("-", "")[:15]
    path = root / f"mock-{stamp}.json"
    path.write_text(json.dumps(asdict(result), indent=1), encoding="utf-8")
    return path


def load_all(root: Path) -> list[MockResult]:
    """Every harvested mock, for calibration."""
    out = []
    for path in sorted(Path(root).glob("mock-*.json")):
        try:
            out.append(MockResult(**json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, TypeError):
            continue
    return out
