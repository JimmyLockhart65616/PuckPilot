"""The live roster: who is on the team today, where they are slotted, and who is hurt.

Nothing in the codebase held this. `DraftBoard.roster(seat)` is the closest, and
it cannot be reused - it is snake-draft shaped, terminates, and has no notion of
an add, a drop, a bench or an injury.

This does NOT go through `yahoo.session.flatten`. That helper merges a whole
Yahoo response into one flat dict, first value wins, which is right for league
metadata and actively wrong here: a roster entry nests `is_keeper: {status,
cost, kept}` and `eligible_positions_to_add: [{position: ...}]` alongside the
player's own `status` and `selected_position.position`, so flattening reports a
keeper as injured and reads a slot off the wrong object. Measured on the real
2026-09-18 roster: all three of Jimmy's keepers came back `status=True` and
every player's selected slot was wrong. Parsed structurally instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from puckpilot.season.settings import BENCH_SLOTS, IR_SLOTS, YAHOO_TO_POS, is_out_status
from puckpilot.yahoo.playermap import _team


class RosterError(RuntimeError):
    """A roster payload could not be read."""


@dataclass(frozen=True)
class RosterPlayer:
    player_key: str
    yahoo_id: str
    name: str
    team: str
    primary_position: str
    yahoo_eligible: frozenset[str]
    selected_slot: str
    nhl_player_id: int | None = None
    status: str = ""
    status_full: str = ""
    injury_note: str = ""
    is_editable: bool = True
    is_undroppable: bool = False
    # Yahoo's `is_keeper`: kept into this season's draft.
    kept: bool = False
    # Next season's keeper standing (`season.keeper_value`), set by the run:
    # his rank among the roster's eligible keepers, and whether that protects
    # him from ever being proposed as a drop.
    keeper_rank: int | None = None
    keeper_protected: bool = False

    @property
    def eligible(self) -> frozenset[str]:
        """Yahoo eligibility as the single-letter positions the engines use.

        Util, IR and IR+ are dropped: they are slots, not positions, and
        `LeagueShape` already carries the util count separately.
        """
        return frozenset(YAHOO_TO_POS[p] for p in self.yahoo_eligible if p in YAHOO_TO_POS)

    @property
    def position(self) -> str:
        """One position, for the paths that still take a single string."""
        return YAHOO_TO_POS.get(self.primary_position, self.primary_position)

    @property
    def is_out(self) -> bool:
        """Cannot play tonight regardless of the schedule. DTD is not out."""
        return is_out_status(self.status)

    @property
    def is_questionable(self) -> bool:
        return bool(self.status) and not self.is_out

    @property
    def on_bench(self) -> bool:
        return self.selected_slot in BENCH_SLOTS

    @property
    def on_ir(self) -> bool:
        return self.selected_slot in IR_SLOTS

    @property
    def starting(self) -> bool:
        return not self.on_bench and not self.on_ir

    def label(self) -> str:
        flag = f" [{self.status}]" if self.status else ""
        return f"{self.name} ({self.team} {self.primary_position}){flag}"


@dataclass(frozen=True)
class TeamRoster:
    league_key: str
    team_key: str
    date: str
    players: tuple[RosterPlayer, ...]
    team_name: str = ""
    is_editable: bool = True
    unmapped: tuple[str, ...] = field(default=())
    # Yahoo's own counters, None when the payload did not carry them. They are
    # the authority: our snapshots record slots, not who actually played, and
    # counting goalie slot-days read the weekly minimum as met by Tuesday.
    goalie_games: int | None = None
    goalie_games_week: int | None = None
    adds_this_week: int | None = None
    adds_week: int | None = None
    moves_season: int | None = None

    def __len__(self) -> int:
        return len(self.players)

    def by_nhl_id(self) -> dict[int, RosterPlayer]:
        return {p.nhl_player_id: p for p in self.players if p.nhl_player_id is not None}

    def starters(self) -> tuple[RosterPlayer, ...]:
        return tuple(p for p in self.players if p.starting)

    def bench(self) -> tuple[RosterPlayer, ...]:
        return tuple(p for p in self.players if p.on_bench)

    def injured(self) -> tuple[RosterPlayer, ...]:
        return tuple(p for p in self.players if p.is_out)

    def illegal_ir(self) -> tuple[RosterPlayer, ...]:
        """Players in an IR slot they are no longer eligible for.

        Yahoo treats that as an illegal roster and refuses every add and drop
        until it is fixed, so anything proposed meanwhile cannot be made.
        """
        return tuple(p for p in self.players if p.on_ir and p.selected_slot not in p.yahoo_eligible)

    def slotted(self) -> dict[str, str]:
        """player_key -> the slot Yahoo currently has him in."""
        return {p.player_key: p.selected_slot for p in self.players}

    def find(self, needle: str) -> RosterPlayer | None:
        low = needle.casefold()
        for p in self.players:
            if p.name.casefold() == low or p.player_key == needle:
                return p
        for p in self.players:
            if low in p.name.casefold():
                return p
        return None


def _fields(parts: list[Any]) -> dict[str, Any]:
    """Merge the single-key dicts of a player's own field list, one level only.

    Deliberately non-recursive: descending is exactly what makes `flatten`
    unsafe on this payload.
    """
    out: dict[str, Any] = {}
    for part in parts:
        if isinstance(part, dict):
            for k, v in part.items():
                out.setdefault(k, v)
    return out


def _positions(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    out = set()
    for item in value:
        if isinstance(item, dict) and "position" in item:
            out.add(str(item["position"]))
        elif isinstance(item, str):
            out.add(item)
    return frozenset(out)


def parse_player(entry: list[Any], player_map: Mapping[str, int] | None = None) -> RosterPlayer:
    """One `player` entry from a roster response.

    Shape: `[[...own fields...], {"selected_position": [...]}, {"is_editable": N}]`
    """
    if not isinstance(entry, list) or not entry:
        raise RosterError(f"unexpected player entry: {type(entry).__name__}")
    core = _fields(entry[0]) if isinstance(entry[0], list) else _fields([entry[0]])
    tail = _fields(entry[1:])

    selected = "?"
    sel = tail.get("selected_position")
    if isinstance(sel, list):
        selected = str(_fields(sel).get("position", "?"))

    key = str(core.get("player_key", ""))
    name = core.get("name") or {}
    keeper = core.get("is_keeper")
    kept = isinstance(keeper, dict) and bool(keeper.get("status") or keeper.get("kept"))
    return RosterPlayer(
        player_key=key,
        yahoo_id=str(core.get("player_id", "")),
        name=str(name.get("full", "")) if isinstance(name, dict) else str(name),
        # Yahoo spells some clubs differently from the NHL ("LA" vs "LAK"),
        # and this field is compared against the NHL schedule - so an
        # unnormalised abbreviation silently reads as "has no game today".
        team=_team(core.get("editorial_team_abbr")),
        primary_position=str(core.get("primary_position", "")),
        yahoo_eligible=_positions(core.get("eligible_positions")),
        selected_slot=selected,
        nhl_player_id=(player_map or {}).get(key),
        # Yahoo omits `status` entirely for a healthy player.
        status=str(core.get("status", "") or ""),
        status_full=str(core.get("status_full", "") or ""),
        injury_note=str(core.get("injury_note", "") or ""),
        is_editable=bool(int(tail.get("is_editable", 1) or 0)),
        is_undroppable=bool(int(core.get("is_undroppable", 0) or 0)),
        kept=kept,
    )


def parse_roster(
    payload: dict,
    team_key: str,
    league_key: str = "",
    player_map: Mapping[str, int] | None = None,
) -> TeamRoster:
    """Build a `TeamRoster` from a raw `/team/{key}/roster` response."""
    try:
        team = payload["fantasy_content"]["team"]
    except (KeyError, TypeError) as exc:
        raise RosterError("not a Yahoo team response") from exc

    node = next((x["roster"] for x in team if isinstance(x, dict) and "roster" in x), None)
    if not isinstance(node, dict):
        raise RosterError(f"no roster node for {team_key}")

    # The team's own name sits in the header section, beside its key - and so
    # do the acquisition counters.
    head = next((x for x in team if isinstance(x, list)), [])
    header = _fields(head) if head else {}
    team_name = str(header.get("name", ""))
    adds = header.get("roster_adds")
    minimum = node.get("minimum_games")

    players_node = node.get("0", {}).get("players")
    players: list[RosterPlayer] = []
    unmapped: list[str] = []
    if isinstance(players_node, dict):
        for i in range(int(players_node.get("count", 0))):
            entry = players_node.get(str(i), {}).get("player")
            if not entry:
                continue
            p = parse_player(entry, player_map)
            players.append(p)
            if p.nhl_player_id is None:
                unmapped.append(p.name)

    return TeamRoster(
        league_key=league_key or team_key.rsplit(".t.", 1)[0],
        team_key=team_key,
        date=str(node.get("date", "")),
        players=tuple(players),
        team_name=team_name,
        is_editable=bool(int(node.get("is_editable", 1) or 0)),
        unmapped=tuple(unmapped),
        goalie_games=_count(minimum, "value"),
        goalie_games_week=_count(minimum, "coverage_value"),
        adds_this_week=_count(adds, "value"),
        adds_week=_count(adds, "coverage_value"),
        moves_season=_int(header.get("number_of_moves")),
    )


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _count(node: Any, key: str) -> int | None:
    """One number out of Yahoo's `{coverage_type, coverage_value, value}` blocks."""
    return _int(node.get(key)) if isinstance(node, dict) else None
