from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from datetime import date

from puckpilot.data import store
from puckpilot.data.moneypuck import KINDS, MoneyPuckClient
from puckpilot.data.nhl import REGULAR_SEASON, NhlApiError, NhlClient

POLITE_DELAY_S = 0.1

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


def season_is_complete(season: str, today: date | None = None) -> bool:
    """'20252026' counts as complete on/after July 1 of its end year (playoffs long over)."""
    end_year = int(season[4:])
    today = today or date.today()
    return today >= date(end_year, 7, 1)


def current_team_abbrevs(nhl: NhlClient) -> list[str]:
    data = nhl.standings_now()
    return sorted({row["teamAbbrev"]["default"] for row in data["standings"]})


def sync_schedules(
    conn: sqlite3.Connection,
    nhl: NhlClient,
    seasons: list[str],
    *,
    delay: float = POLITE_DELAY_S,
    progress: Progress = _noop,
) -> dict[str, int]:
    """Upsert every game visible from the current 32 clubs' season schedules.

    Vanished franchises (e.g. ARI) still appear as opponents on current clubs'
    schedules, so this covers the whole league; dedupe is by game_id.
    Returns unique game counts per season.
    """
    teams = current_team_abbrevs(nhl)
    counts: dict[str, int] = {}
    for season in seasons:
        seen: set[int] = set()
        for team in teams:
            sched = nhl.club_schedule_season(team, season)
            for g in sched.get("games", []):
                gid = g["id"]
                if gid in seen:
                    continue
                seen.add(gid)
                store.upsert_schedule_game(
                    conn,
                    game_id=gid,
                    season=str(g.get("season", season)),
                    game_type=g["gameType"],
                    game_date=g["gameDate"],
                    start_time_utc=g.get("startTimeUTC"),
                    home_team=g["homeTeam"]["abbrev"],
                    away_team=g["awayTeam"]["abbrev"],
                )
            time.sleep(delay)
        conn.commit()
        counts[season] = len(seen)
        progress(f"  {season}: {len(seen)} games")
    return counts


BOXSCORE_GROUPS = ("forwards", "defense", "goalies")


BoxscoreRow = tuple[int, int, str, str | None, str]


def boxscore_rows(bs: dict, game_id: int, season: str) -> list[BoxscoreRow]:
    """Flatten a boxscore's playerByGameStats into upsert tuples.

    Hits and blocked shots appear only here — the player game-log endpoint omits
    both, so these rows are what make the HIT/BLK categories scoreable.
    """
    pbg = bs.get("playerByGameStats") or {}
    out: list[tuple[int, int, str, str | None, str]] = []
    for side in ("awayTeam", "homeTeam"):
        team = (bs.get(side) or {}).get("abbrev")
        for group in BOXSCORE_GROUPS:
            for p in (pbg.get(side) or {}).get(group, []):
                pid = p.get("playerId")
                if pid is None:
                    continue
                out.append((game_id, int(pid), season, team, json.dumps(p)))
    return out


def sync_boxscores(
    conn: sqlite3.Connection,
    nhl: NhlClient,
    seasons: list[str],
    *,
    delay: float = POLITE_DELAY_S,
    today: date | None = None,
    progress: Progress = _noop,
) -> dict[str, dict[str, int]]:
    """Fetch per-game boxscores for regular-season games already played.

    Incremental via sync_meta key 'boxscore:{game_id}', mirroring the
    'gamelog:{pid}:{season}' convention. Unplayed games are skipped rather than
    stored empty, so re-running after games are played picks them up.
    """
    today = today or date.today()
    cutoff = today.isoformat()
    report: dict[str, dict[str, int]] = {}
    for season in seasons:
        games = conn.execute(
            "SELECT game_id, game_date FROM nhl_schedule"
            " WHERE season = ? AND game_type = ? AND game_date < ?"
            " ORDER BY game_date, game_id",
            (season, REGULAR_SEASON, cutoff),
        ).fetchall()
        synced = skipped = empty = rows_written = 0
        for i, g in enumerate(games, start=1):
            gid = int(g["game_id"])
            key = f"boxscore:{gid}"
            if store.get_meta(conn, key) == "done":
                skipped += 1
                continue
            rows = boxscore_rows(nhl.boxscore(gid), gid, season)
            if not rows:
                empty += 1
                continue
            store.upsert_boxscore_rows(conn, rows)
            store.set_meta(conn, key, "done")
            rows_written += len(rows)
            synced += 1
            if synced % 50 == 0:
                conn.commit()
                progress(f"    {season}: {i}/{len(games)} games")
            time.sleep(delay)
        conn.commit()
        progress(
            f"  {season}: {synced} boxscores synced, {skipped} already done, "
            f"{empty} without player stats, {rows_written} player rows"
        )
        report[season] = {
            "games": len(games),
            "synced": synced,
            "skipped": skipped,
            "empty": empty,
            "rows": rows_written,
        }
    return report


ROSTER_GROUPS = ("forwards", "defensemen", "goalies")


def sync_player_bios(
    conn: sqlite3.Connection,
    nhl: NhlClient,
    seasons: list[str],
    *,
    delay: float = POLITE_DELAY_S,
    progress: Progress = _noop,
) -> int:
    """Birth dates etc. from team rosters - 32 requests per season, not one per player.

    Walking several seasons picks up players who have since retired or changed
    teams; later seasons overwrite earlier ones, which is fine for static bio data.
    """
    teams = current_team_abbrevs(nhl)
    seen: set[int] = set()
    for season in seasons:
        for team in teams:
            try:
                data = nhl.roster(team, season)
            except NhlApiError:
                continue  # franchise did not exist that season
            for group in ROSTER_GROUPS:
                for p in data.get(group, []):
                    pid = p.get("id")
                    if pid is None:
                        continue
                    store.upsert_player_bio(
                        conn,
                        player_id=int(pid),
                        birth_date=p.get("birthDate"),
                        height_in=p.get("heightInInches"),
                        weight_lb=p.get("weightInPounds"),
                        shoots=p.get("shootsCatches"),
                    )
                    seen.add(int(pid))
            time.sleep(delay)
        conn.commit()
        progress(f"  {season}: {len(seen)} players with bio data so far")
    return len(seen)


def sync_current_rosters(
    conn: sqlite3.Connection,
    nhl: NhlClient,
    season: str,
    *,
    delay: float = POLITE_DELAY_S,
    progress: Progress = _noop,
) -> dict[str, int]:
    """Point `nhl_players.team_abbrev` at who each player actually plays for.

    MoneyPuck is the player-discovery mechanism and it is synced per season with
    INSERT OR REPLACE, so whichever season ran last decided every player's team.
    That left 436 players who appeared in 2025-26 carrying their 2022-23 club.

    It is not cosmetic. `projections._team_win_rate_by_goalie` groups on this
    column and `GOALIE_TEAM_WIN_BLEND` is 0.5, so half of every goalie's
    projected wins - a scored category - came from the wrong team's record.
    Re-projected with correct teams the mean absolute change was 2.24 wins and
    the largest was 13.19.

    32 requests, one per club, same shape as `sync_player_bios`.
    """
    teams = current_team_abbrevs(nhl)
    seen = changed = 0
    for team in teams:
        try:
            data = nhl.roster(team, season)
        except NhlApiError:
            progress(f"  {team}: no roster for {season}")
            continue
        for group in ROSTER_GROUPS:
            for p in data.get(group, []):
                pid = p.get("id")
                if pid is None:
                    continue
                seen += 1
                changed += store.update_player_team(conn, int(pid), team)
        time.sleep(delay)
    conn.commit()
    progress(f"  {len(teams)} rosters, {seen} players, {changed} team(s) corrected")
    return {"teams": len(teams), "players": seen, "changed": changed}


def sync_players_and_logs(
    conn: sqlite3.Connection,
    nhl: NhlClient,
    mp: MoneyPuckClient,
    seasons: list[str],
    *,
    with_logs: bool = True,
    delay: float = POLITE_DELAY_S,
    progress: Progress = _noop,
) -> dict[str, dict[str, int]]:
    """MoneyPuck season CSVs -> nhl_players + mp_season_stats, then per-player NHL game logs.

    Player discovery comes from the situation='all' rows (everyone with >=1 game).
    All situations are stored in mp_season_stats (PP/SH splits matter in Phase 2).
    Game-log sync is incremental: sync_meta key 'gamelog:{player_id}:{season}' is set
    to 'done' once a completed season's log is stored, and such players are skipped
    on re-runs. Current-season logs are always refetched.
    """
    report: dict[str, dict[str, int]] = {}
    for season in seasons:
        players: dict[int, dict] = {}
        stat_rows = 0
        for kind in KINDS:
            rows = mp.season_csv(season, kind, refresh=not season_is_complete(season))
            if rows is None:
                progress(f"  {season}: no MoneyPuck data yet, skipping")
                break
            singular = kind[:-1]
            for row in rows:
                pid = int(row["playerId"])
                store.upsert_mp_season_stat(
                    conn,
                    player_id=pid,
                    season=season,
                    kind=singular,
                    situation=row["situation"],
                    name=row["name"],
                    team=row.get("team"),
                    position=row.get("position"),
                    stats_json=json.dumps(row),
                )
                stat_rows += 1
                if row["situation"] == "all":
                    players[pid] = row
        if not players:
            report[season] = {"players": 0, "stat_rows": 0, "logs_synced": 0, "logs_skipped": 0}
            continue
        for pid, row in players.items():
            store.upsert_player(conn, pid, row["name"], row.get("position"), row.get("team"))
        conn.commit()
        progress(f"  {season}: {len(players)} players, {stat_rows} MoneyPuck stat rows")

        synced = skipped = 0
        if with_logs:
            complete = season_is_complete(season)
            for i, pid in enumerate(sorted(players), start=1):
                key = f"gamelog:{pid}:{season}"
                if complete and store.get_meta(conn, key) == "done":
                    skipped += 1
                    continue
                log = nhl.player_game_log(pid, season)
                for entry in log.get("gameLog", []):
                    store.upsert_game_log(
                        conn,
                        player_id=pid,
                        game_id=entry["gameId"],
                        season=season,
                        game_type=REGULAR_SEASON,
                        game_date=entry["gameDate"],
                        team_abbrev=entry.get("teamAbbrev"),
                        opponent_abbrev=entry.get("opponentAbbrev"),
                        is_home=1 if entry.get("homeRoadFlag") == "H" else 0,
                        stats_json=json.dumps(entry),
                    )
                if complete:
                    store.set_meta(conn, key, "done")
                conn.commit()
                synced += 1
                if i % 100 == 0:
                    progress(f"    {season}: {i}/{len(players)} players processed")
                time.sleep(delay)
            progress(f"  {season}: game logs synced for {synced}, skipped {skipped} (already done)")
        report[season] = {
            "players": len(players),
            "stat_rows": stat_rows,
            "logs_synced": synced,
            "logs_skipped": skipped,
        }
    return report
