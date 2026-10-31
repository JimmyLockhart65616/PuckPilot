"""Drop tiers: which of a manager's players a move may let go, and on what terms.

Every add search, make-room move and release picks its drop from the same
players (`legality.droppable`), and until 2026-10-31 the only say a manager
had in which of them was the keeper rule. Two rules decided the rest, and the
add gate measured them: the `stream_spots` players worth least over the rest
of the season rotate - any move that pays this week may drop one - and anyone
else goes only for a player worth more over the season, which is an upgrade,
not streaming. A tier makes each of those something a manager can see and
set, player by player:

    never     never a drop - not for a streamer, an upgrade, or to make room
    upgrade   only for a player worth more over the rest of the season: by at
              least `upgrade_margin` of his value (0: by anything at all)
    rotation  for any move that pays this week - and, where the manager has
              given standing approval (`season/standing.py`), made after a
              window to stop it instead of a tap

Where a player's tier comes from, first match wins:

    1. the platform marks him undroppable: never, and no pin moves him
    2. the manager's pin (the page's roster, or `ppilot season tiers --set`)
    3. one of next season's protected keepers (`keeper_value`): never
    4. a rotation spot: pinned rotation players take them first, and the
       rest go to the cheapest unpinned players who may be dropped today
    5. everyone else: upgrade

With nothing pinned that is exactly the rule the gate measured: a tier changes
nothing until a manager sets one.

A tier is not a status. A player out, NA, on IR, or already the drop of an
approved move is not a drop while that lasts, whatever his tier - those are
facts about tonight, not the manager's view of the player.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum


class Tier(StrEnum):
    """A manager's terms for letting a player go. The value is what `player_tiers` stores."""

    NEVER = "never"
    UPGRADE = "upgrade"
    ROTATION = "rotation"

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {Tier.NEVER: "Never", Tier.UPGRADE: "Upgrade only", Tier.ROTATION: "Rotation"}

# Where a player's tier came from (`Placement.source`). AUTO is also what a
# manager sends to take a pin off: back to the rule.
YAHOO = "yahoo"
YOU = "you"
KEEPER = "keeper"
AUTO = "auto"


@dataclass(frozen=True)
class Placement:
    """One player's tier, where it came from, and why - in the page's words."""

    tier: Tier
    source: str
    why: str

    def payload(self) -> dict:
        return {
            "tier": self.tier.value,
            "label": self.tier.label,
            "source": self.source,
            "why": self.why,
        }


def tier_of(value: str | None) -> Tier | None:
    """A typed or stored tier; None is the automatic rule. An unknown one is refused."""
    if not value or value == AUTO:
        return None
    try:
        return Tier(value)
    except ValueError as e:
        known = ", ".join([*(t.value for t in Tier), AUTO])
        raise ValueError(f"unknown tier {value!r} (known: {known})") from e


def pinned(p) -> Tier | None:
    """The manager's pin on `p` (`RosterPlayer.tier_pin`), if he has one."""
    value = getattr(p, "tier_pin", "") or ""
    return Tier(value) if value in _VALUES else None


_VALUES = frozenset(t.value for t in Tier)


def is_never(p) -> bool:
    """Never a drop: the platform says so, the manager pinned him there, or
    he is one of next season's protected keepers and nobody pinned otherwise."""
    if getattr(p, "is_undroppable", False):
        return True
    pin = pinned(p)
    if pin is not None:
        return pin is Tier.NEVER
    return bool(getattr(p, "keeper_protected", False))


def clears(incoming: float, worth: float, margin: float = 0.0) -> bool:
    """Whether a player worth `incoming` over the rest of the season is upgrade
    enough to let go one worth `worth`: more by at least `margin` of his value
    - of its size, so a player valued below zero is not made easier to drop.
    Margin 0 is the rule before tiers: worth more, by anything."""
    return incoming >= worth + margin * abs(worth)


def _fixed(p) -> Placement | None:
    """A tier that needs no values: the platform's, the manager's, a keeper's."""
    if getattr(p, "is_undroppable", False):
        return Placement(Tier.NEVER, YAHOO, "Yahoo does not let him be dropped")
    pin = pinned(p)
    if pin is not None:
        return Placement(pin, YOU, "set by you")
    if getattr(p, "keeper_protected", False):
        rank = getattr(p, "keeper_rank", None)
        return Placement(
            Tier.NEVER,
            KEEPER,
            "one of next season's keepers" + (f" (rank {rank})" if rank else ""),
        )
    return None


def _upgrade_why(margin: float) -> str:
    more = f"at least {margin:.0%} more" if margin > 0 else "more"
    return f"goes only for a player worth {more} over the rest of the season"


def board(
    players: Iterable,
    keep: Mapping[str, float],
    stream_spots: int,
    droppable: Iterable | None = None,
    upgrade_margin: float = 0.0,
) -> dict[str, Placement]:
    """Every player's tier, by player key.

    `keep` is each player's value over the rest of the season - what letting
    him go costs (`week.keep_values`) - and `droppable` the players a move may
    let go today (`legality.droppable`; all of `players` if None). Only those
    can take an automatic rotation spot: a player out tonight is not streamed,
    and his spot goes to the next cheapest, as it always has. A tie at the
    last spot's value takes a spot too, as the stream floor always did.
    """
    players = list(players)
    pool = {p.player_key for p in (players if droppable is None else droppable)}
    out: dict[str, Placement] = {}
    for p in players:
        placed = _fixed(p)
        if placed is not None:
            out[p.player_key] = placed
    taken = sum(1 for k in pool if k in out and out[k].tier is Tier.ROTATION)
    left = stream_spots - taken
    open_values = sorted(
        float(keep.get(p.player_key, 0.0))
        for p in players
        if p.player_key in pool and p.player_key not in out
    )
    floor = open_values[min(left, len(open_values)) - 1] if open_values and left > 0 else None
    for p in players:
        if p.player_key in out:
            continue
        worth = float(keep.get(p.player_key, 0.0))
        if floor is not None and p.player_key in pool and worth <= floor:
            out[p.player_key] = Placement(
                Tier.ROTATION,
                AUTO,
                f"one of the {stream_spots} worth least over the rest of the season",
            )
        else:
            out[p.player_key] = Placement(Tier.UPGRADE, AUTO, _upgrade_why(upgrade_margin))
    return out


# -- the manager's pins ------------------------------------------------------


def pins(conn: sqlite3.Connection, manager: str, league_key: str) -> dict[str, Tier]:
    """player key -> the tier this manager pinned him to."""
    rows = conn.execute(
        "SELECT player_key, tier FROM player_tiers WHERE manager = ? AND league_key = ?",
        (manager, league_key),
    )
    return {key: Tier(tier) for key, tier in rows if tier in _VALUES}


def listing(conn: sqlite3.Connection, manager: str, league_key: str) -> list[dict]:
    """Every pin, with who it is and when it was set - for the CLI."""
    rows = conn.execute(
        "SELECT player_key, tier, name, nhl_player_id, set_at, set_via FROM player_tiers "
        "WHERE manager = ? AND league_key = ? ORDER BY name",
        (manager, league_key),
    )
    keys = ("player_key", "tier", "name", "nhl_player_id", "set_at", "set_via")
    return [dict(zip(keys, r, strict=True)) for r in rows]


def pin(
    conn: sqlite3.Connection,
    manager: str,
    league_key: str,
    player_key: str,
    tier: Tier | None,
    *,
    name: str = "",
    nhl_player_id: int | None = None,
    via: str = "cli",
    at: str | None = None,
) -> None:
    """Set a player's tier, or with None take the pin off (back to the rule).

    Kept when he leaves the roster - it is the manager's view of the player -
    and applied again if he comes back.
    """
    if tier is None:
        conn.execute(
            "DELETE FROM player_tiers WHERE manager = ? AND league_key = ? AND player_key = ?",
            (manager, league_key, player_key),
        )
    else:
        conn.execute(
            "INSERT OR REPLACE INTO player_tiers (manager, league_key, player_key, tier, name, "
            "nhl_player_id, set_at, set_via) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                manager,
                league_key,
                player_key,
                Tier(tier).value,
                name,
                nhl_player_id,
                at or datetime.now(UTC).isoformat(timespec="seconds"),
                via,
            ),
        )
    conn.commit()


def annotate(roster, pinned_tiers: Mapping[str, Tier]):
    """The roster with each player's pin on him (`RosterPlayer.tier_pin`), and
    none on a player whose pin was taken off."""

    def with_pin(p):
        want = pinned_tiers[p.player_key].value if p.player_key in pinned_tiers else ""
        return p if getattr(p, "tier_pin", "") == want else replace(p, tier_pin=want)

    return replace(roster, players=tuple(with_pin(p) for p in roster.players))
