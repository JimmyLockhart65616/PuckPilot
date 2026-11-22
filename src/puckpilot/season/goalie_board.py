"""Every club's goalies at once: who starts, how often, and whose they are.

The add search sees a free-agent goalie as a line of numbers; a manager sees a
net. This is the net, league-wide, from the same start source the run prices
with (announced calls in front of the history model when they are used):
tonight's P(start), Yahoo's call and tag, the starts expected over the rest of
this week and over the regular season after it (his share of his club's starts,
`goalies.GoalieWorkload`, times his club's games left), and whose roster he is
on. The season figure is not this week's P(start) averaged: two days left of a
week, one the back half of a back-to-back, would read a backup as a starter.

Every run keeps one row a goalie a day (`goalie_forecasts`), so the forecast
is scored against what happened (`check`) rather than trusted.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class GoalieRow:
    player_id: int
    name: str
    club: str
    tonight: float | None  # None: his club does not play tonight
    week_starts: float
    season_starts: float
    call: bool | None
    status: str
    owner: str  # a team's name, or "free agent"


def _clubs(conn, season: str) -> dict[int, str]:
    """goalie -> his club: the latest game this season he dressed for or played.

    The start model knows a club's goalies from its boxscores, which list both
    dressed goalies, so a backup who has sat every game is still his club's;
    the game logs cover a game whose boxscore is not in yet. Game logs alone
    leave that backup clubless, with no season starts beside his week's.
    """
    from puckpilot.season.goalies import _dressed

    latest: dict[int, tuple[str, str]] = {}

    def seen(pid: int, date: str, team: str) -> None:
        if team and (pid not in latest or date > latest[pid][0]):
            latest[pid] = (date, team)

    for pid, date, team in conn.execute(
        "SELECT player_id, game_date, team_abbrev FROM nhl_game_logs WHERE season = ? "
        "AND json_extract(stats_json, '$.gamesStarted') IS NOT NULL",
        (season,),
    ):
        seen(int(pid), date, team)
    for (date, team), pids in _dressed(conn, season).items():
        for pid in pids:
            seen(pid, date, team)
    return {pid: team for pid, (_date, team) in latest.items()}


def owners(conn, league_key: str, day: str) -> dict[int, str]:
    """NHL id -> the name of the fantasy team that holds him (latest snapshot)."""
    names = dict(
        conn.execute(
            "SELECT team_key, name FROM league_week_stats WHERE league_key = ? GROUP BY team_key",
            (league_key,),
        ).fetchall()
    )
    out: dict[int, str] = {}
    for team, latest in conn.execute(
        "SELECT team_key, MAX(date) FROM yahoo_roster_snapshots WHERE league_key = ? "
        "AND date <= ? GROUP BY team_key",
        (league_key, day),
    ).fetchall():
        for (pid,) in conn.execute(
            "SELECT nhl_player_id FROM yahoo_roster_snapshots WHERE team_key = ? AND date = ? "
            "AND nhl_player_id IS NOT NULL",
            (team, latest),
        ):
            out[pid] = names.get(team, team)
    return out


def board(
    conn,
    runtime,
    week,
    day: str,
    source,
    holders: dict[int, str],
    priors: dict[int, float] | None = None,
) -> list[GoalieRow]:
    """Every goalie the start source gives a start this week or tonight.

    `priors` (`goalies.projected_shares`) are the preseason shares a goalie's
    share so far is shrunk toward; without them it is his share so far.
    """
    from puckpilot.season.goalies import GoalieWorkload
    from puckpilot.season.outlook import remaining_days
    from puckpilot.season.starters import latest_calls

    season = runtime.nhl_season
    club = _clubs(conn, season)
    workload = GoalieWorkload(conn, season, priors=priors)
    names = dict(conn.execute("SELECT player_id, full_name FROM nhl_players"))
    days = [d for d in week.dates() if d >= day]
    tonight = source.starts(day) if days else {}
    per: dict[int, list[float]] = {}
    for d in days:
        for pid, p in source.starts(d).items():
            per.setdefault(pid, []).append(float(p))
    games_left: dict[str, int] = {}
    from puckpilot.season import calendar

    for d in remaining_days(runtime, week):
        for c in calendar.teams_playing(conn, d, season):
            games_left[c] = games_left.get(c, 0) + 1
    try:
        calls = latest_calls(conn, day)
    except sqlite3.Error:
        calls = {}
    rows = []
    for pid, ps in per.items():
        c = club.get(pid, "")
        share = workload.share(pid, day)
        if share is None:  # never dressed this season and not projected
            share = sum(ps) / len(ps)
        _team, call, status = calls.get(pid, ("", None, ""))
        rows.append(
            GoalieRow(
                player_id=pid,
                name=names.get(pid, str(pid)),
                club=c,
                tonight=float(tonight[pid]) if pid in tonight else None,
                week_starts=sum(ps),
                season_starts=share * games_left.get(c, 0),
                call=call,
                status=status,
                owner=holders.get(pid, "free agent"),
            )
        )
    return sorted(rows, key=lambda r: (r.club, -r.week_starts))


def save(conn, rows: list[GoalieRow], day: str, week_end: str, made_at: str | None = None) -> int:
    """Keep this run's forecast: one row a goalie a day (the latest run wins)."""
    at = made_at or datetime.now(UTC).isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR REPLACE INTO goalie_forecasts (date, player_id, club, tonight, week_starts,"
        " week_end, season_starts, owner, made_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                day,
                r.player_id,
                r.club,
                r.tonight,
                r.week_starts,
                week_end,
                r.season_starts,
                r.owner,
                at,
            )
            for r in rows
        ],  # fmt: skip
    )
    conn.commit()
    return len(rows)


def text(rows: list[GoalieRow], day: str) -> str:
    out = [
        f"Every club's goalies, {day}: tonight's chance to start, Yahoo's call, starts expected",
        "over the rest of this week and of the season, and whose he is.",
        "",
        "  club  goalie                    tonight  call    week  season  owner",
    ]
    for r in rows:
        call = {True: "starts", False: "sits", None: ""}[r.call]
        tag = f" [{r.status}]" if r.status else ""
        t = "  -  " if r.tonight is None else f"{r.tonight:5.0%}"
        out.append(
            f"  {r.club:4}  {(r.name + tag)[:24]:24}  {t}    {call:6}  {r.week_starts:4.1f}"
            f"  {r.season_starts:6.1f}  {r.owner}"
        )
    return "\n".join(out)


def check(conn, since: str = "", before: str = "") -> str:
    """Each kept forecast of the rest of a week against the starts made in it.

    Only weeks over by `before` (default: every kept week whose end has a
    game log after it) are scored; a week still being played is not.
    """
    last = conn.execute("SELECT MAX(game_date) FROM nhl_game_logs").fetchone()[0] or ""
    cutoff = before or last
    rows = conn.execute(
        "SELECT date, player_id, week_starts, week_end FROM goalie_forecasts "
        "WHERE date >= ? AND week_end < ?",
        (since, cutoff),
    ).fetchall()
    if not rows:
        return "No finished week's goalie forecast kept yet."
    err = []
    for date, pid, want, end in rows:
        made = conn.execute(
            "SELECT COUNT(*) FROM nhl_game_logs WHERE player_id = ? AND game_date BETWEEN ? AND ? "
            "AND json_extract(stats_json, '$.gamesStarted') = 1",
            (pid, date, end),
        ).fetchone()[0]
        err.append(made - want)
    n = len(err)
    return (
        f"Goalie forecasts scored: {n} goalie-days. Starts made - forecast, rest of the week: "
        f"mean {sum(err) / n:+.2f}, mean absolute {sum(abs(e) for e in err) / n:.2f}."
    )
