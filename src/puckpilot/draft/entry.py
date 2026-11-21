"""Picks entered by hand: the second pick source on draft night.

The websocket feed measured 190/190 against a real draft, and that is exactly
why this exists: a feed that works every time fails silently the one time it
does not, and the only way to keep the clock right then is a human who can see
the room. The web view and the terminal console both call these, so "taken"
cannot mean one thing on one screen and something else on the other.

Every function returns `(ok, message)` rather than raising. A refusal on a
30-second clock has to read as a sentence ("Cale Makar is already off the
board"), not a traceback - and a repeated entry, which is what a human under
pressure produces, is an answer, not an error.
"""

from __future__ import annotations

from puckpilot.draft.board import DraftBoard, DraftBoardError
from puckpilot.keepers import _norm

HAND_LABEL = "(entered by hand)"


def resolve_player(board: DraftBoard, text: str) -> tuple[int | None, str]:
    """A player id from an id or a typed name; `(None, why)` when it cannot.

    The same rules `ManualFeed.submit` uses: an exact name wins outright, an
    ambiguous one is refused with the candidates rather than guessed at.
    """
    text = (text or "").strip()
    if not text:
        return None, "no player given"
    if text.lstrip("-").isdigit():
        pid = int(text)
        if pid not in board._row_of:
            return None, f"player {pid} is not on this board"
        return pid, ""
    rows = board.find(text, limit=6)
    if not rows:
        gone = board.find(text, limit=1, available_only=False)
        if gone:
            return None, f"{board.u.names[gone[0]]} is already off the board"
        return None, f"no available player matches {text!r}"
    exact = [r for r in rows if _norm(str(board.u.names[r])) == _norm(text)]
    if len(exact) == 1:
        rows = exact
    if len(rows) > 1:
        names = ", ".join(str(board.u.names[r]) for r in rows[:5])
        return None, f"{text!r} matches several players: {names}"
    return int(board.u.ids[rows[0]]), ""


def mark_taken(board: DraftBoard, text: str, seat: int | None = None) -> tuple[bool, str]:
    """Drafted - by the seat on the clock unless one is given. Uses a pick.

    If the feed later delivers the same pick it is rejected as already off the
    board (see `feed.apply`), so entering a pick ahead of a slow feed never
    counts it twice.
    """
    pid, why = resolve_player(board, text)
    if pid is None:
        return False, why
    try:
        pick = board.record(pid, seat, "manual")
    except DraftBoardError as e:
        return False, str(e)
    return True, f"{pick.name} taken by seat {pick.seat} (pick {pick.overall + 1})"


def mark_unknown(board: DraftBoard, seat: int | None = None) -> tuple[bool, str]:
    """Advance the clock one pick without naming anyone."""
    try:
        pick = board.record_unknown(seat, HAND_LABEL)
    except DraftBoardError as e:
        return False, str(e)
    return True, f"pick {pick.overall + 1} (seat {pick.seat}) recorded as unknown"


def mark_kept(board: DraftBoard, text: str, seat: int | None) -> tuple[bool, str]:
    """A keeper nobody declared: off the board WITHOUT using a pick."""
    if seat is None:
        return False, "a keeper belongs to a team: give the seat that kept him"
    pid, why = resolve_player(board, text)
    if pid is None:
        return False, why
    try:
        pick = board.add_keeper(pid, seat)
    except DraftBoardError as e:
        return False, str(e)
    return True, f"{pick.name} marked as kept by seat {pick.seat} (no pick used)"


def unmark_kept(board: DraftBoard, text: str) -> tuple[bool, str]:
    """Put a keeper back on the board."""
    text = (text or "").strip()
    pid: int | None
    if text.lstrip("-").isdigit():
        pid = int(text)
    else:
        keepers = [p for p in board.keeper_picks if _norm(p.name) == _norm(text)]
        pid = keepers[0].player_id if len(keepers) == 1 else None
    if pid is None:
        return False, f"{text!r} is not a keeper on this board"
    pick = board.remove_keeper(pid)
    if pick is None:
        return False, f"{text!r} is not a keeper on this board"
    return True, f"{pick.name} is back on the board"
