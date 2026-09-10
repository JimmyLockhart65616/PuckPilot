"""Mock drafts with a real outcome attached.

A Yahoo mock draft tells you nothing about whether you drafted well — it ends
and that is that. This ends by replaying the roster you just built through a
real NHL season and reporting where it finished, so a mock is a measurement
rather than a rehearsal.

That makes it the practice loop, and it needs no Yahoo access at all: the board,
the opposing field and the season are all local. It is deliberately the same
walk-forward setup the draft sim is scored on (project from seasons before the
target, draft, replay the target's real game logs), so a result here is
comparable to the recorded top-3 rate rather than a different game.

`--auto` hands your seat to the engine instead of prompting, which doubles as
the regression gate: the live console must reproduce `RosterValuePolicy`.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from puckpilot.draft.advice import format_board, recommend
from puckpilot.draft.board import DraftBoard, DraftBoardError, Pick
from puckpilot.draft.engine import RosterValuePolicy
from puckpilot.draft.feed import ManualFeed, SimFeed, apply
from puckpilot.draft.h2h import run_h2h_season
from puckpilot.draft.replay import G_WIDTH, build_replay_data, replay_roster
from puckpilot.draft.sim import _default_opponents, build_universe, keepers_for
from puckpilot.league import DEFAULT_LEAGUE, LeagueConfig

HELP = """\
  <enter>      take the engine's top pick
  1-15         take that numbered candidate
  <name>       take a player by name (partial is fine)
  u / undo     take back the last pick
  ?            this help
  q            abandon the draft
"""


@dataclass
class MockResult:
    seat: int
    roster: list[Pick]
    finish: int
    seed: int
    record: tuple[int, int, int]
    cat_share: dict[str, float]
    pick_latency_ms: list[float] = field(default_factory=list)
    text: str = ""


def _grade(
    board: DraftBoard,
    conn: sqlite3.Connection,
    league: LeagueConfig,
    target_season: str,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Replay every seat's roster through the real season and play the H2H year."""
    u = board.u
    shape = board.rules.shape
    skater_keys = [c.key for c in league.skater_cats]
    data = build_replay_data(conn, target_season, skater_keys)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))
    scalar = dict(zip(u.ids.tolist(), u.z_total.tolist(), strict=True))

    n_weeks = max(data.n_weeks, 1)
    sk = np.zeros((shape.n_teams, n_weeks, len(skater_keys)))
    g = np.zeros((shape.n_teams, n_weeks, G_WIDTH))
    for seat in range(shape.n_teams):
        ids = [p.player_id for p in board.roster(seat)]
        sk[seat], g[seat] = replay_roster(ids, positions, scalar, data, shape)

    result = run_h2h_season(
        sk,
        g,
        league.skater_cats,
        league.goalie_cats,
        skater_keys,
        regular_weeks=league.regular_weeks,
        playoff_teams=league.playoff_teams,
        playoff_weeks=league.playoff_weeks,
    )
    return result, sk, g


def run_mock(
    conn: sqlite3.Connection,
    league: LeagueConfig = DEFAULT_LEAGUE,
    seat: int = 0,
    seed: int | None = None,
    auto: bool = False,
    target_season: str = "20252026",
    train_seasons: tuple[str, ...] = ("20242025", "20232024", "20222023"),
    prompt: Callable[[str], str] = input,
    progress: Callable[[str], None] = print,
) -> MockResult:
    rng = np.random.default_rng(seed)
    rules = league.draft_rules()
    policy = RosterValuePolicy()

    progress("Building the board (once; every pick after this is arithmetic)...")
    t0 = time.perf_counter()
    universe = build_universe(conn, target_season, train_seasons, league)
    keepers = keepers_for(conn, universe, target_season, league, rng, warn=progress)
    board = DraftBoard(universe, rules, my_seat=seat, keepers=keepers)
    progress(f"  ready in {time.perf_counter() - t0:.1f}s — {len(board.slots)} live picks\n")
    if board.unmatched_keepers:
        progress(f"  WARNING: {len(board.unmatched_keepers)} keepers could not be placed\n")

    # The engine occupies our seat only in --auto; otherwise the human does.
    opponents = _default_opponents(rng, league)
    order = rng.permutation(len(opponents))
    bots, oi = [], 0
    for s in range(rules.shape.n_teams):
        if s == seat:
            bots.append(policy)
        else:
            bots.append(opponents[order[oi]])
            oi += 1

    sim_feed = SimFeed(bots, rng, skip_seats=set() if auto else {seat})
    manual = ManualFeed()
    latency: list[float] = []
    abandoned = False

    stalled = 0
    while not board.complete:
        on_clock = board.on_the_clock()
        if on_clock != seat or auto:
            accepted, _rejected = apply(board, sim_feed.poll(board))
            # A feed that stops producing usable picks must end the draft rather
            # than spin: the board can only run dry if the universe is too small
            # for the roster, and that is a setup error worth surfacing.
            stalled = 0 if accepted else stalled + 1
            if stalled > rules.shape.n_teams:
                progress("  no further picks available — ending the draft early")
                break
            continue
        stalled = 0

        t = time.perf_counter()
        candidates = recommend(board, policy, n=15)
        latency.append((time.perf_counter() - t) * 1000)

        progress("")
        progress(format_board(board, candidates))
        progress("")
        try:
            answer = prompt("Pick (enter = engine's choice, ? for help): ").strip()
        except (EOFError, KeyboardInterrupt):
            abandoned = True
            break

        if answer in {"q", "quit"}:
            abandoned = True
            break
        if answer == "?":
            progress(HELP)
            continue
        if answer in {"u", "undo"}:
            undone = board.undo()
            progress(f"  undid: {undone.name}" if undone else "  nothing to undo")
            continue
        if not answer:
            board.record(candidates[0].player_id, source="engine")
            progress(f"  took {candidates[0].name}")
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            chosen = candidates[int(answer) - 1]
            board.record(chosen.player_id, source="manual")
            progress(f"  took {chosen.name}")
            continue
        try:
            manual.submit(board, answer)
            accepted, rejected = apply(board, manual.poll(board))
            for pick in accepted:
                progress(f"  took {pick.name}")
            for message in rejected:
                progress(f"  {message}")
        except DraftBoardError as e:
            progress(f"  {e}")

    if abandoned:
        return MockResult(seat, board.roster(seat), 0, 0, (0, 0, 0), {}, latency, "abandoned")

    progress("\nDraft complete. Replaying the real season...")
    result, sk, _g = _grade(board, conn, league, target_season)
    finish = int(result.finish[seat])
    record = tuple(int(x) for x in result.records[seat])
    cat_rec = result.cat_records[seat]
    total_cats = max(1, int(cat_rec.sum()))
    cat_share = {"category win rate": float(cat_rec[0]) / total_cats}

    lines = [
        "",
        f"Mock draft — {league.name}, seat {seat}, replayed over {target_season}",
        "=" * 64,
        f"Finish:        {finish} of {rules.shape.n_teams}" + ("  CHAMPION" if finish == 1 else ""),
        f"Regular season: {record[0]}-{record[1]}-{record[2]} (seed {int(result.seeds[seat])})",
        f"Category win rate: {cat_share['category win rate']:.1%}",
        "",
        "Your roster:",
    ]
    for pick in sorted(board.roster(seat), key=lambda p: (p.overall < 0, p.overall)):
        tag = "keeper" if pick.overall < 0 else f"#{pick.overall + 1}"
        lines.append(f"  {tag:>6}  [{pick.position}] {pick.name}")
    if latency:
        lines += [
            "",
            f"Recommendation latency: median {np.median(latency):.1f}ms, "
            f"max {max(latency):.1f}ms over {len(latency)} picks",
        ]
    lines += [
        "",
        "One draft is one sample: a single finish is not an expected finish.",
        "Use `ppilot draft sim` for a rate over many drafts.",
    ]

    return MockResult(
        seat=seat,
        roster=board.roster(seat),
        finish=finish,
        seed=int(result.seeds[seat]),
        record=record,
        cat_share=cat_share,
        pick_latency_ms=latency,
        text="\n".join(lines),
    )
