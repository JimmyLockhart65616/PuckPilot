"""The draft-night console: what to take, right now.

This is the thing the rest of the draft code exists to serve. It holds the board,
watches picks arrive, and keeps a ranked shortlist on screen the whole time.

Two design rules, both load-bearing:

- **Nothing model-shaped is on the clock.** The board is built once (~8s) and
  every recommendation after that is numpy over arrays, measured at well under a
  millisecond. A 30-second pick timer is never at risk.
- **The engine never picks.** It shows a shortlist with the reasoning on both
  sides and a human decides. On draft night that is the whole point; the engine
  only drafts for itself in mock runs.

Input runs on its own thread so the screen keeps refreshing while you type: with
an automatic feed running, picks land and the shortlist updates without you
touching anything.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from puckpilot.draft import entry
from puckpilot.draft.advice import can_wait_on, market_watchlist, recommend
from puckpilot.draft.board import DraftBoard
from puckpilot.draft.engine import RosterValuePolicy, Universe
from puckpilot.draft.feed import apply

HELP = """\
Commands (type and press enter):
  t NAME [@N]   taken: NAME was drafted (by seat N, default the seat on the clock)
  x [@N]        unknown pick: the room took someone - advance the clock one pick
  k NAME @N     kept: a keeper nobody declared, held by seat N - uses NO pick
  unk NAME      put a keeper back on the board
  u / undo      take back the last pick, however it was entered
  seat N        change which seat is yours
  ?             this help
  q             quit
Picks normally arrive on their own from the feed. Type them only when the
"behind the room" warning says the feed has missed some.

What the columns mean, first time here:
  VORP     value over the last startable player at that position - the one
           number safe to compare ACROSS positions (a C and a D at the same
           VORP are equally valuable picks). Higher is always better.
  ADP      average draft position: where the room actually takes him.
  Lasts?   odds he is still there at your next turn. High = safe to wait on
           him; low = take him now or lose him.
  *        marks a player who fills a starting slot right now, not the bench.
"Likely still there next turn" names the room is expected to leave for you.
"Market only, no projection" is real draft buzz (usually a rookie) priced from
the room, not from us - a pick of one is recorded, but never recommended.
"""


@dataclass
class LiveConfig:
    top: int = 12
    refresh: float = 1.0
    show_survivors: bool = True


def _clear() -> str:
    # ANSI home+clear: keeps the console in one place instead of scrolling away.
    return "\033[H\033[J"


def feed_health(board: DraftBoard, feed) -> list[str]:
    """Lines that say whether the board can be trusted right now.

    The same facts the web view shows. Before this the terminal read only
    `feed.last_error`, so a board that had simply stopped hearing the room -
    no error, no gaps - looked healthy while every survival number was stale.
    """
    lines = list(getattr(board, "warnings", []) or [])
    if feed is None or not hasattr(feed, "status"):
        return lines
    try:
        st = feed.status()
    except Exception as e:  # a status call must never take the console down
        return [*lines, f"feed status failed: {e.__class__.__name__}"]
    room = int(st.get("room_picks") or st.get("highest_pick") or 0)
    drift = board.drift(room) if room else 0
    if drift > 0:
        lines.append(
            f"!! BOARD IS {drift} PICK{'S' if drift != 1 else ''} BEHIND THE ROOM "
            "- enter them (t NAME / x) or every 'lasts' figure is stale"
        )
    if st.get("gaps"):
        lines.append("missing pick numbers: " + ", ".join(str(g) for g in st["gaps"]))
    names = st.get("unmapped_names") or []
    if names:
        lines.append("not on our board (each used a pick): " + ", ".join(names[-5:]))
    return lines


def _seat_arg(text: str) -> tuple[str, int | None, str]:
    """'Cale Makar @4' -> ('Cale Makar', 4, ''). The seat marker is '@' so a
    name containing a number is never read as one."""
    if "@" not in text:
        return text.strip(), None, ""
    name, _, raw = text.rpartition("@")
    try:
        return name.strip(), int(raw.strip()), ""
    except ValueError:
        return name.strip(), None, f"seat after @ must be a number, got {raw.strip()!r}"


def handle_command(board: DraftBoard, text: str) -> str | None:
    """Apply one hand-entry command. Returns the status line, or None when
    `text` is not a hand-entry command at all."""
    head, _, rest = text.partition(" ")
    head = head.lower()
    if head in {"t", "taken"}:
        name, seat, err = _seat_arg(rest)
        if err:
            return err
        if seat is not None and not 0 <= seat < board.n_teams:
            return f"seat {seat} outside 0..{board.n_teams - 1}"
        return entry.mark_taken(board, name, seat)[1]
    if head in {"x", "unknown"}:
        _, seat, err = _seat_arg(rest)
        if err:
            return err
        if seat is not None and not 0 <= seat < board.n_teams:
            return f"seat {seat} outside 0..{board.n_teams - 1}"
        return entry.mark_unknown(board, seat)[1]
    if head in {"k", "kept"}:
        name, seat, err = _seat_arg(rest)
        if err:
            return err
        if seat is not None and not 0 <= seat < board.n_teams:
            return f"seat {seat} outside 0..{board.n_teams - 1}"
        return entry.mark_kept(board, name, seat)[1]
    if head in {"unk", "unkeep"}:
        return entry.unmark_kept(board, rest)[1]
    return None


def render(
    board: DraftBoard, cands, cfg: LiveConfig, status: str = "", health: list[str] | None = None
) -> str:
    seat = board.my_seat
    on_clock = board.on_the_clock()
    mine = on_clock == seat
    rnd = board.current_round()
    nxt = board.next_pick_no(seat)

    lines = [_clear()]
    for h in health or []:
        lines.append(f"[!] {h}")
    banner = ">>> YOUR PICK <<<" if mine else f"seat {on_clock} on the clock"
    lines.append(f"Round {rnd}   pick {board.made + 1}/{len(board.slots)}   {banner}")
    if nxt is not None:
        away = nxt - board.made
        lines.append(f"Your next pick: #{nxt + 1}" + ("" if mine else f"  ({away} picks away)"))
    needs = board.needs(seat)
    if needs:
        lines.append("Still to fill: " + ", ".join(f"{p}x{n}" for p, n in sorted(needs.items())))
    lines.append("")

    lines.append(
        f"{'#':>2}  {'Player':<24}{'Pos':<5}{'Tm':<5}{'VORP':>6}{'Score':>7}{'ADP':>6}{'Lasts?':>8}"
    )
    lines.append("-" * 66)
    for i, c in enumerate(cands[: cfg.top], start=1):
        star = "*" if c.fills_starter else " "
        lasts = f"{c.p_survive:.0%}"
        lines.append(
            f"{i:>2}{star} {c.name:<24}{c.position:<5}{c.team:<5}"
            f"{c.vorp:>6.2f}{c.score:>7.2f}{c.adp_rank:>6.0f}{lasts:>8}"
        )
    lines.append("")

    if cfg.show_survivors and cands:
        # The single most useful thing on a clock: who you can afford to wait
        # on. One definition of "lasts", shared with the reasons on each card.
        likely = can_wait_on(board, cands[: cfg.top])
        if likely:
            lines.append("Likely still there next turn: " + ", ".join(likely[:5]))

    # Priced from the room, not from us - see draft.market. Never part of the
    # ranked shortlist above; shown separately so a real pick of one of these
    # can still be recognized instead of landing as an unknown player.
    watch = market_watchlist(board, seat, n=5)
    if watch:
        named = ", ".join(f"{c.name} (~{c.adp_rank:.0f})" for c in watch)
        lines.append(f"Market only, no projection: {named}")

    roster = board.roster(seat)
    if roster:
        by_pos: dict[str, list[str]] = {}
        for p in roster:
            by_pos.setdefault(p.position, []).append(p.name.split()[-1])
        lines.append(
            "Your roster: "
            + "  ".join(f"{pos}: {', '.join(v)}" for pos, v in sorted(by_pos.items()))
        )
    recent = board.picks[-6:]
    if recent:
        lines.append("Recent: " + " | ".join(f"{p.name} (s{p.seat})" for p in recent))
    lines.append("")
    if status:
        lines.append(status)
    lines.append("> (? for help)")
    return "\n".join(lines)


def _stdin_thread(q: queue.Queue) -> None:
    while True:
        try:
            q.put(input())
        except (EOFError, KeyboardInterrupt):
            q.put("q")
            return


def run_live(
    board: DraftBoard,
    feed=None,
    cfg: LiveConfig | None = None,
    policy: RosterValuePolicy | None = None,
    out: Callable[[str], None] = print,
    pump: Callable[[float], None] | None = None,
) -> DraftBoard:
    """Drive the console until the draft ends or the user quits.

    `pump` is required when a Playwright-backed feed is attached: the sync
    driver only dispatches events from inside a Playwright call, so a loop that
    merely waits on stdin never sees a new tab open and the draft room hangs on
    a blank page.
    """
    cfg = cfg or LiveConfig()
    policy = policy or RosterValuePolicy()
    inbox: queue.Queue = queue.Queue()
    threading.Thread(target=_stdin_thread, args=(inbox,), daemon=True).start()

    status = "Ready."
    dirty = True
    cands: list = []
    last_health: list[str] = []

    while not board.complete:
        # 1. automatic picks, if a feed is attached
        if feed is not None:
            events = feed.poll(board)
            if events:
                accepted, rejected = apply(board, events)
                if accepted:
                    status = "feed: " + ", ".join(p.name for p in accepted[-3:])
                    dirty = True
                if rejected:
                    status += f"  ({len(rejected)} rejected)"
            err = getattr(feed, "last_error", None)
            if err:
                status = f"feed error: {err}"

        # 2. redraw - also whenever the health lines change, so a feed that
        # goes quiet is announced without waiting for the next pick
        health = feed_health(board, feed)
        if health != last_health:
            dirty = True
            last_health = health
        if dirty:
            cands = recommend(board, policy, n=max(cfg.top, 15))
            out(render(board, cands, cfg, status, health))
            dirty = False

        # 3. input, without blocking the refresh loop. When a browser feed is
        # attached the wait must happen *inside* Playwright, or its events never
        # dispatch; stdin is then checked without blocking.
        if pump is not None:
            pump(cfg.refresh)
            try:
                line = inbox.get_nowait()
            except queue.Empty:
                continue
        else:
            try:
                line = inbox.get(timeout=cfg.refresh)
            except queue.Empty:
                continue

        text = line.strip()
        if text in {"q", "quit"}:
            break
        if text == "?":
            out(HELP)
            continue
        if text in {"u", "undo"}:
            undone = board.undo()
            status = f"undid {undone.name}" if undone else "nothing to undo"
            dirty = True
            continue
        handled = handle_command(board, text)
        if handled is not None:
            status = handled
            dirty = True
            continue
        if text.startswith("seat "):
            try:
                board.my_seat = int(text.split()[1])
                status = f"your seat is now {board.my_seat}"
            except (ValueError, IndexError):
                status = "usage: seat N"
            dirty = True
            continue
        dirty = True

    closing = "Draft complete." if board.complete else "Stopped (board kept)."
    out(render(board, recommend(board, policy, n=cfg.top), cfg, closing, feed_health(board, feed)))
    return board


def attach_market_frame(
    universe: Universe, market_frame: pd.DataFrame, adp: dict[int, int] | None
) -> Universe:
    """Concat `market_frame` onto `universe` and rebuild, or hand `universe`
    back unchanged if there is nothing to add.

    Pulled out of `build_live_board` so the one part of this feature with any
    real mechanics - reconciling a `with_adp`-only array update against the
    frame it never touched, and recomputing `has_market` after a sort scrambles
    row order - can be tested against a small synthetic `Universe` rather than
    only through a live database.
    """
    if market_frame.empty:
        return universe
    # `.frame` was never touched by `with_adp` - only the array was - so the
    # real ADP must be re-stamped onto it before it becomes the base of a new
    # Universe, or every existing player's ADP would silently revert to the
    # pre-market proxy.
    combined = pd.concat(
        [universe.frame.assign(adp_rank=universe.adp_rank), market_frame]
    ).sort_values("vorp", ascending=False)
    out = Universe(combined)
    # Positional concatenation of the old `has_market` array would be wrong
    # here: `sort_values` just reordered every row, so membership has to be
    # recomputed by id, not carried along by position.
    market_priced = (set(adp) if adp else set()) | set(market_frame.index)
    out.has_market = np.array([int(pid) in market_priced for pid in out.ids], dtype=bool)
    return out


def build_live_board(
    conn,
    league,
    seat: int,
    season: str = "20262027",
    train_seasons: tuple[str, ...] = ("20252026", "20242025", "20232024"),
    adp: dict[int, int] | None = None,
    league_key: str | None = None,
    mock_glob: str = "data/mocks/*.json",
    progress: Callable[[str], None] = print,
    seed: int = 0,
) -> DraftBoard:
    """The one slow step, done before the draft rather than during it.

    Keepers are the part that used to be missing, and their absence was not
    cosmetic: without them the kept players stay on the board and top the
    shortlist, our roster reads empty so the engine mis-states what we still
    need, ADP is never re-based, and the slot count is the full roster rather
    than the picks that will actually happen - which makes `next_pick_no`, and
    therefore every survival probability, wrong.

    `league_key` additionally pulls in market-implied rows for players with a
    Yahoo ADP but no NHL history at all (rookies, mainly) - see `draft.market`.
    Without it those players are simply invisible: not just unranked, but
    absent from the board, so a real pick of one of them cannot be recorded and
    the pick clock drifts exactly the way an unknown player used to. Requires
    `adp` too, since the market curve is fit against real Yahoo ADP.
    """
    from puckpilot.draft.sim import build_universe, keepers_for

    t0 = time.perf_counter()
    progress("Building board...")
    # Facts about the build that change what every number on screen means.
    # Printed here AND carried on the board, because the drafter is looking at
    # the web view, not at the scrollback of the terminal that built it.
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg.strip())
        progress(msg)

    # Anyone the market prices goes on the board even if our own ranking would
    # have cut him: the room can draft him, and a pick we cannot record is a
    # pick our clock does not see.
    universe = build_universe(
        conn, season, train_seasons, league, market_ids=set(adp) if adp else None
    )
    if adp:
        # Real Yahoo ADP beats the prior-season-value proxy, and survival_discount
        # reads directly off it.
        ranks = np.array(
            [float(adp.get(int(pid), len(universe) + 1)) for pid in universe.ids],
            dtype=float,
        )
        universe = universe.with_adp(ranks)
        # Who the market actually prices, captured HERE and not inferred later:
        # un-priced players are given a sentinel rank past the end, and
        # `effective_adp` re-ranks everyone into 1..n when keepers come off,
        # erasing the sentinel. After that there is no way to tell a genuinely
        # cheap player from one the market never mentioned.
        universe.has_market = np.array([int(pid) in adp for pid in universe.ids], dtype=bool)
        progress(f"  using Yahoo ADP for {sum(1 for p in universe.ids if int(p) in adp)} players")

        if league_key:
            from puckpilot.draft.market import build_market_frame

            market_frame = build_market_frame(
                conn, league_key, universe, mock_glob=mock_glob, progress=progress
            )
            universe = attach_market_frame(universe, market_frame, adp)

        # Yahoo's own position eligibility, after the market rows are in so
        # they get it too. Shown on every row; used for roster accounting only
        # when the policy runs multi-position.
        from puckpilot.yahoo.playermap import load_eligibility

        universe = universe.with_eligibility(load_eligibility(conn, league_key))
    else:
        # This used to be silence: `--yahoo` given bare, or a key with a typo,
        # built a board on the proxy and said nothing at all.
        warn(
            "  WARNING: no Yahoo ADP - every 'lasts %' and the survival discount are "
            "computed against a proxy (last season's value order), not the room's ADP"
        )

    # Deterministic: a live board rebuilt mid-draft must not re-deal keepers.
    keepers = keepers_for(conn, universe, season, league, np.random.default_rng(seed), warn=warn)
    board = DraftBoard(
        universe,
        league.draft_rules(),
        my_seat=seat,
        keepers=keepers,
        keeper_placement=league.keeper_placement,
    )
    board.adp_source = "yahoo" if adp else "proxy"

    kept = sum(len(v) for v in keepers.values())
    progress(f"  {kept} keepers off the board; {len(board.slots)} live picks remain")
    if board.unmatched_keepers:
        # An unplaced keeper leaves an elite player wrongly draftable - the worst
        # board error there is, so it is stated rather than logged and forgotten.
        warn(f"  WARNING: {len(board.unmatched_keepers)} keeper(s) could not be placed")
    slots = league.n_keepers * league.shape.n_teams
    declared = len(league.keepers_for_season(season))
    if league.n_keepers and declared < slots:
        # A keeper missing from the list is still on this board, and a kept star
        # tops the shortlist exactly like an available one. Nothing downstream
        # can tell the difference - only the drafter can, if told.
        warn(
            f"  WARNING: {declared} keepers listed for {season} against {slots} keeper "
            "slots - any undeclared keeper is still shown as available (use 'kept' to "
            "strike one)"
        )
    owners = league.keeper_owners_for_season(season)
    if seat not in owners:
        warn(
            f"  NOTE: no declared keepers for seat {seat}; ownership is dealt evenly, "
            "so your roster panel is a guess. Set [keepers.owners] in the league file."
        )
    elif league.n_keepers and len(board.roster(seat)) < league.n_keepers:
        warn(
            f"  NOTE: seat {seat} has {len(board.roster(seat))} of {league.n_keepers} keepers "
            "declared, so the roster panel and picks left assume the rest are live picks"
        )
    board.warnings = warnings
    progress(f"  ready in {time.perf_counter() - t0:.1f}s\n")
    return board
