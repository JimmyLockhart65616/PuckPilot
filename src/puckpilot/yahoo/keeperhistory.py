"""Keeper contracts, reconstructed from the league's own Yahoo draft history.

A keeper sheet kept by hand drifts: the one this was first run against omitted
two players kept for two straight seasons and listed one who was re-drafted
fresh. Yahoo's record
does not drift. A player kept going into season S shows up in S's draft results
as a pick by the manager who held him at the end of S-1, in the keeper rounds.
Walk that back through the league's `renew` chain and every contract falls out:
who holds it, how many times it has been used, and whether it has run out.

What this can NOT know is a first-year keep - a player drafted last season whom
his manager chooses to keep for the first time. That is a decision, not a
record. So those are listed as candidates for the manager's open slots, never
declared.

Read-only, and the output is a suggestion to paste: the league file is private
and hand-maintained, and nothing here writes it.

The Yahoo shapes are confined to `fetch_history`; `derive_contracts` works on
plain data so it can be tested without a browser.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def bare(player_key: str) -> str:
    """'465.p.6743' -> '6743'. Yahoo player ids are stable across seasons; only
    the game prefix changes."""
    return str(player_key).rsplit(".", 1)[-1]


def renewed_from(meta: dict) -> str | None:
    """The previous season's league key from `league_meta`'s `renew` field
    ('465_12345' -> '465.l.12345'), or None for a league's first season."""
    raw = str(meta.get("renew") or "")
    if "_" not in raw:
        return None
    game, league = raw.split("_", 1)
    return f"{game}.l.{league}"


@dataclass
class SeasonHistory:
    league_key: str
    season: str  # Yahoo's season year, e.g. "2025" for 2025-26
    # team_key -> {"guid", "nickname", "name"}
    teams: dict[str, dict]
    # each {"round", "pick", "team_key", "player_key"}
    picks: list[dict]
    # team_key -> bare player ids on that team at the end of the season
    rosters: dict[str, list[str]]
    # bare player id -> name, from rosters (and anything else seen)
    names: dict[str, str] = field(default_factory=dict)

    def guid_of(self, team_key: str) -> str:
        return str(self.teams.get(team_key, {}).get("guid") or team_key)

    def final_holder(self) -> dict[str, str]:
        """bare player id -> guid of the manager holding him at season end."""
        out = {}
        for team_key, ids in self.rosters.items():
            for pid in ids:
                out[pid] = self.guid_of(team_key)
        return out


def _round(p: dict) -> int | None:
    try:
        return int(p.get("round"))
    except (TypeError, ValueError):
        return None


# A final round counts as a keeper round when MORE than this share of its picks
# are players the picking manager already held. Real keeper rounds sit near
# 100% (a team keeping fewer than the maximum makes one live pick there); an
# ordinary late round sits near 0%, with the odd re-draft of last year's player.
KEEPER_ROUND_SHARE = 0.5


def keeper_window(season: SeasonHistory, previous: SeasonHistory | None, default: int) -> int:
    """How many final rounds held keepers in `season`, read from the picks.

    Measured rather than assumed because leagues change their keeper count,
    and a fixed window one round too wide reads a live late-round re-draft of
    last year's player as a keeper - which happened on the first real run.
    """
    if previous is None:
        return default
    held = previous.final_holder()
    by_round: dict[int, list[bool]] = {}
    for p in season.picks:
        rnd = _round(p)
        if rnd is None:
            continue
        pid = bare(p.get("player_key", ""))
        by_round.setdefault(rnd, []).append(held.get(pid) == season.guid_of(p.get("team_key", "")))
    window = 0
    for rnd in sorted(by_round, reverse=True):
        flags = by_round[rnd]
        if sum(flags) / len(flags) <= KEEPER_ROUND_SHARE:
            break
        window += 1
    return window


def keeper_picks(
    season: SeasonHistory,
    previous: SeasonHistory | None,
    roster_rounds: int,
    window: int,
) -> dict[str, str]:
    """bare player id -> guid, for the picks in `season` that were keepers.

    A keeper pick is in the last `window` rounds AND is a player the same
    manager held at the end of the previous season. The second condition is
    what separates a keeper from a live pick inside a keeper round (a team
    keeping fewer than the maximum); without a previous season to check
    against, only the round window is applied.
    """
    held = previous.final_holder() if previous else None
    out: dict[str, str] = {}
    for p in season.picks:
        rnd = _round(p)
        if rnd is None or rnd <= roster_rounds - window:
            continue
        pid = bare(p.get("player_key", ""))
        guid = season.guid_of(p.get("team_key", ""))
        if held is not None and held.get(pid) != guid:
            continue
        out[pid] = guid
    return out


@dataclass
class ManagerContracts:
    guid: str
    nickname: str
    team_name: str
    current_team_key: str | None
    continuing: list[tuple[str, int]] = field(default_factory=list)  # (bare id, times kept)
    expired: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)  # first-year keep options

    def open_slots(self, n_keepers: int) -> int:
        return max(0, n_keepers - len(self.continuing))


@dataclass
class ContractReport:
    seasons: list[str]  # league keys used, newest first
    managers: list[ManagerContracts]
    names: dict[str, str]
    # kept last season but on nobody's roster at its end
    lapsed: list[str] = field(default_factory=list)
    # league key -> how many final rounds held keepers that season
    keeper_rounds: dict[str, int] = field(default_factory=dict)


def derive_contracts(
    history: list[SeasonHistory],
    current_teams: dict[str, dict],
    n_keepers: int,
    keeper_years: int,
    roster_rounds: int,
    window: int | None = None,
) -> ContractReport:
    """Contracts going into the season after `history[-1]`.

    `history` runs oldest to newest and ends with the most recent COMPLETED
    season. `window` pins how many final rounds held keepers; by default it is
    measured per season (`keeper_window`), falling back to `n_keepers` for the
    oldest season, which has nothing before it to measure against.
    """
    kept: list[dict[str, str]] = []
    self_windows: list[int] = []
    for i, season in enumerate(history):
        prev = history[i - 1] if i > 0 else None
        w = window if window is not None else keeper_window(season, prev, n_keepers)
        self_windows.append(w)
        kept.append(keeper_picks(season, prev, roster_rounds, w))

    names: dict[str, str] = {}
    for season in history:
        names.update(season.names)

    def times_kept(pid: str) -> int:
        n = 0
        for season_kept in reversed(kept):
            if pid not in season_kept:
                break
            n += 1
        return n

    latest = history[-1]
    by_guid_current = {str(t.get("guid")): k for k, t in current_teams.items()}
    managers: dict[str, ManagerContracts] = {}
    for team_key, ids in latest.rosters.items():
        guid = latest.guid_of(team_key)
        team = current_teams.get(by_guid_current.get(guid, ""), latest.teams.get(team_key, {}))
        m = managers.setdefault(
            guid,
            ManagerContracts(
                guid=guid,
                nickname=str(team.get("nickname") or latest.teams[team_key].get("nickname")),
                team_name=str(team.get("name") or ""),
                current_team_key=by_guid_current.get(guid),
            ),
        )
        for pid in ids:
            n = times_kept(pid)
            if n == 0:
                m.candidates.append(pid)
            elif n >= keeper_years:
                m.expired.append(pid)
            else:
                m.continuing.append((pid, n))

    held = latest.final_holder()
    lapsed = [pid for pid in kept[-1] if pid not in held]
    return ContractReport(
        seasons=[s.league_key for s in reversed(history)],
        managers=sorted(managers.values(), key=lambda m: m.nickname.lower()),
        names=names,
        lapsed=lapsed,
        keeper_rounds=dict(zip([s.league_key for s in history], self_windows, strict=True)),
    )


def fetch_history(
    session,
    league_key: str,
    depth: int,
    progress: Progress = _noop,
) -> tuple[list[SeasonHistory], dict[str, dict], dict]:
    """(completed seasons oldest->newest, this season's teams, this season's meta).

    `depth` completed seasons are fetched: keeper_years + 1 is enough to tell
    an expired contract from a live one.
    """
    meta = session.league_meta(league_key)
    current = {t["team_key"]: t for t in session.teams(league_key)}
    history: list[SeasonHistory] = []
    key = renewed_from(meta)
    while key and len(history) < depth:
        progress(f"  reading {key}")
        m = session.league_meta(key)
        teams = {t["team_key"]: t for t in session.teams(key)}
        rosters: dict[str, list[str]] = {}
        names: dict[str, str] = {}
        for team_key in teams:
            players = session.roster(team_key)
            rosters[team_key] = [bare(p.get("player_key", "")) for p in players]
            for p in players:
                names[bare(p.get("player_key", ""))] = str(p.get("full") or "")
        history.append(
            SeasonHistory(
                league_key=key,
                season=str(m.get("season") or ""),
                teams=teams,
                picks=session.draft_results(key),
                rosters=rosters,
                names=names,
            )
        )
        key = renewed_from(m)
    history.reverse()
    return history, current, meta
