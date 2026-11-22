"""A real morning's add search, run again from what was saved that day.

The gates judge a knob over replayed seasons; this answers the narrower
question a manager asks - "would it have suggested something else on the day
it suggested that?" Everything is as of the morning: our roster and the
opponent's as saved that day, the free agents as read that day, values and
the start model clamped to the day before, and the manager's own search terms.
Only the knobs passed in differ. Reported, never tuned to: three mornings are
anecdotes, not evidence.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta


def replay(
    conn,
    runtime,
    manager,
    day: str,
    *,
    season_weight: float = 0.0,
    their_stream: bool = False,
    pool_size: int = 150,
    terms=None,
    season_keep: bool = False,
    keeper_weight: float = 0.0,
):
    """The week plan the search makes on `day` with these knobs, or None.

    `terms` replaces the manager's `authority.transactions` (a goalie lane, a
    drop choice); None: his own."""
    from puckpilot.season import streams as streams_mod
    from puckpilot.season.add_gate import AsOfValues
    from puckpilot.season.g3 import saved_roster
    from puckpilot.season.goalies import AsOfGoalieSource, TrailingStartShareSource
    from puckpilot.season.odds import OddsModel
    from puckpilot.season.pool import load_pool
    from puckpilot.season.run import _train_seasons, goalie_terms, rival_rosters, week_for
    from puckpilot.season.values import build_value_model
    from puckpilot.season.week import build_week_plan

    week = week_for(runtime, day)
    lk = runtime.league_key
    team = manager.team_key or _our_team(conn, manager.name, lk, day)
    if week is None or not team:
        return None
    row = conn.execute(
        "SELECT opponent_key FROM league_week_stats WHERE league_key = ? AND week = ? "
        "AND team_key = ?",
        (lk, week.number, team),
    ).fetchone()
    ours = saved_roster(conn, lk, team, day)
    theirs = saved_roster(conn, lk, row[0], day) if row else None
    if ours is None or theirs is None:
        return None
    season = runtime.nhl_season
    train = _train_seasons(season)
    cutoff = (_date.fromisoformat(day) - timedelta(days=1)).isoformat()
    base = build_value_model(conn, season, train, manager.league)
    keepers = None
    if keeper_weight > 0 and manager.league.n_keepers > 0:
        from pathlib import Path

        from puckpilot.config import Settings
        from puckpilot.season import keeper_value as kv

        path = Settings()._resolve(Path("data")) / f"keepers-{season}.json"
        keepers = kv.board(conn, manager.league, season, cutoff, path)
    values = AsOfValues(base, cutoff, keepers=keepers)
    goalies = AsOfGoalieSource(TrailingStartShareSource(conn, season, fallback_season=train[0]),
                               cutoff)  # fmt: skip
    pool = load_pool(conn, lk, day)[:pool_size]
    stream = None
    if their_stream:
        played = [w for w in runtime.weeks if w.end < week.start]
        added = streams_mod.added_by_week(conn, lk, season, played)
        teams = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT team_key FROM league_week_stats WHERE league_key = ?", (lk,)
            )
        ]
        stream = streams_mod.measure(added, [w.number for w in played], teams, team=row[0])
    terms = terms or manager.authority.transactions
    return build_week_plan(
        conn,
        runtime,
        manager.league,
        week,
        row[0],
        ours,
        theirs,
        pool,
        base.frame,
        goalies,
        values,
        min_gain=terms.min_weekly_gain,
        odds_model=OddsModel(),
        add_scoring=terms.add_scoring,
        min_expected_gain=terms.min_expected_gain,
        playoff_reserve=terms.playoff_reserve,
        stream_spots=terms.stream_spots,
        upgrade_margin=terms.upgrade_margin,
        screen_by=terms.screen,
        from_day=day,
        their_stream=stream,
        season_rivals=rival_rosters(conn, lk, team, day) if season_weight else None,
        season_weight=season_weight,
        season_keep=season_keep,
        keeper_weight=keeper_weight,
        **goalie_terms(terms, runtime, week, day, ours, theirs),
    )


def _our_team(conn, manager: str, league_key: str, day: str) -> str:
    row = conn.execute(
        "SELECT team_key FROM yahoo_roster_snapshots WHERE manager = ? AND league_key = ? "
        "AND date <= ? ORDER BY date DESC LIMIT 1",
        (manager, league_key, day),
    ).fetchone()
    return row[0] if row else ""


def lines(plan) -> list[str]:
    """The targets as a card would head them: add, drop, and the price's parts."""
    if plan is None:
        return ["  (no saved rosters for that day)"]
    if not plan.targets:
        return ["  no move clears the bar"]
    out = []
    for t in plan.targets:
        drop = t.drop.name if t.drop is not None else "an open spot"
        parts = (
            f" (week {t.week_gain:+.2f}, season {t.season_gain:+.2f})"
            if t.season_gain is not None and t.week_gain is not None
            else ""
        )
        gain = t.gain if t.gain is not None else t.score
        out.append(f"  add {t.player.name} ({t.player.position}) for {drop}: {gain:+.2f}{parts}")
    return out
