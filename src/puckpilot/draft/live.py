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

from puckpilot.draft.advice import recommend
from puckpilot.draft.board import DraftBoard
from puckpilot.draft.engine import RosterValuePolicy
from puckpilot.draft.feed import apply

HELP = """\
Commands (type and press enter):
  u / undo      take back the last pick the feed recorded
  seat N        change which seat is yours
  ?             this help
  q             quit
Picks arrive on their own from the websocket feed; there is nothing to type in.
"""


@dataclass
class LiveConfig:
    top: int = 12
    refresh: float = 1.0
    show_survivors: bool = True


def _clear() -> str:
    # ANSI home+clear: keeps the console in one place instead of scrolling away.
    return "\033[H\033[J"


def render(board: DraftBoard, cands, cfg: LiveConfig, status: str = "") -> str:
    seat = board.my_seat
    on_clock = board.on_the_clock()
    mine = on_clock == seat
    rnd = board.current_round()
    nxt = board.next_pick_no(seat)

    lines = [_clear()]
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
        # The single most useful thing on a clock: who you can afford to wait on.
        likely = [c.name for c in cands[: cfg.top] if c.p_survive >= 0.65]
        if likely:
            lines.append("Likely still there next turn: " + ", ".join(likely[:5]))

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

        # 2. redraw
        if dirty:
            cands = recommend(board, policy, n=max(cfg.top, 15))
            out(render(board, cands, cfg, status))
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
            # The feed is the only pick source, so undo is the sole recovery
            # hatch if a frame ever lands wrong. It is not manual entry.
            undone = board.undo()
            status = f"undid {undone.name}" if undone else "nothing to undo"
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
    out(render(board, recommend(board, policy, n=cfg.top), cfg, closing))
    return board


def build_live_board(
    conn,
    league,
    seat: int,
    season: str = "20262027",
    train_seasons: tuple[str, ...] = ("20252026", "20242025", "20232024"),
    adp: dict[int, int] | None = None,
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
    """
    from puckpilot.draft.sim import build_universe, keepers_for

    t0 = time.perf_counter()
    progress("Building board...")
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
        progress(f"  using Yahoo ADP for {sum(1 for p in universe.ids if int(p) in adp)} players")

    # Deterministic: a live board rebuilt mid-draft must not re-deal keepers.
    keepers = keepers_for(
        conn, universe, season, league, np.random.default_rng(seed), warn=progress
    )
    board = DraftBoard(universe, league.draft_rules(), my_seat=seat, keepers=keepers)

    kept = sum(len(v) for v in keepers.values())
    progress(f"  {kept} keepers off the board; {len(board.slots)} live picks remain")
    if board.unmatched_keepers:
        # An unplaced keeper leaves an elite player wrongly draftable - the worst
        # board error there is, so it is stated rather than logged and forgotten.
        progress(f"  WARNING: {len(board.unmatched_keepers)} keeper(s) could not be placed")
    owners = league.keeper_owners_for_season(season)
    if seat not in owners:
        progress(
            f"  NOTE: no declared keepers for seat {seat}; ownership is dealt evenly, "
            "so your roster panel is a guess. Set [keepers.owners] in the league file."
        )
    progress(f"  ready in {time.perf_counter() - t0:.1f}s\n")
    return board
