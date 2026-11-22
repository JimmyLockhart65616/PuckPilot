"""Every club's goalies at once, and the forecast kept and scored."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from puckpilot.data import store
from puckpilot.season import goalie_board
from puckpilot.season.goalies import StaticGoalieSource
from tests.conftest import add_goalie_game, add_player

S = "20262027"


def _games(db, dates, home="AAA", away="BBB"):
    for gid, d in enumerate(dates, 1):
        store.upsert_schedule_game(
            db, game_id=gid, season=S, game_type=2, game_date=d, start_time_utc=None,
            home_team=home, away_team=away,
        )  # fmt: skip
    db.commit()


def _week(n, start, end, dates):
    return SimpleNamespace(number=n, start=start, end=end, dates=lambda: list(dates))


def test_the_board_gives_each_goalie_his_starts_ahead_and_whose_he_is(db):
    add_player(db, 1, "Starter One", "G")
    add_player(db, 2, "Backup Two", "G")
    add_goalie_game(db, 1, S, 900, date="2026-11-01")
    add_goalie_game(db, 2, S, 901, date="2026-11-02")
    this = ["2026-11-16", "2026-11-18"]
    later = ["2026-11-24", "2026-11-26", "2026-11-28"]
    _games(db, this + later)
    w1 = _week(1, "2026-11-16", "2026-11-22", this + ["2026-11-20", "2026-11-22"])
    w2 = _week(2, "2026-11-23", "2026-11-29", later)
    rt = SimpleNamespace(nhl_season=S, weeks=[w1, w2], playoff_start_week=0)
    source = StaticGoalieSource({d: {1: 0.75, 2: 0.25} for d in this})
    rows = goalie_board.board(db, rt, w1, "2026-11-16", source, {1: "Team One"})
    by = {r.player_id: r for r in rows}
    assert by[1].tonight == 0.75 and by[1].week_starts == pytest.approx(1.5)
    assert by[1].season_starts == pytest.approx(0.75 * 3)  # his share x his club's 3 left
    assert (by[1].owner, by[2].owner) == ("Team One", "free agent")
    assert "Starter One" in goalie_board.text(rows, "2026-11-16")


def test_a_backup_who_has_only_dressed_is_still_his_clubs(db):
    import json

    add_player(db, 1, "Starter One", "G")
    add_player(db, 3, "Dressed Three", "G")
    add_goalie_game(db, 1, S, 1, date="2026-11-01")  # AAA, and 3 sat on the bench
    db.execute(
        "INSERT INTO nhl_boxscore_stats (game_id, player_id, season, team_abbrev, stats_json)"
        " VALUES (1, 3, ?, 'AAA', ?)",
        (S, json.dumps({"playerId": 3, "position": "G"})),
    )
    this = ["2026-11-16"]
    later = ["2026-11-24", "2026-11-26"]
    _games(db, ["2026-11-01", *this, *later])
    w1 = _week(1, "2026-11-16", "2026-11-22", this)
    w2 = _week(2, "2026-11-23", "2026-11-29", later)
    rt = SimpleNamespace(nhl_season=S, weeks=[w1, w2], playoff_start_week=0)
    # Tonight the back half of a back-to-back: he is likely to start it, but
    # his season is his share of AAA's starts (0 of 1 so far), not tonight's.
    source = StaticGoalieSource({"2026-11-16": {1: 0.1, 3: 0.9}})

    def by(priors=None):
        rows = goalie_board.board(db, rt, w1, "2026-11-16", source, {}, priors=priors)
        return {r.player_id: r for r in rows}

    assert by()[3].club == "AAA" and by()[3].week_starts == pytest.approx(0.9)
    assert by()[3].season_starts == 0.0
    shrunk = (0 + 10 * 0.2) / (1 + 10)  # goalies.WORKLOAD_PRIOR_GAMES of his projection
    assert by({3: 0.2})[3].season_starts == pytest.approx(shrunk * 2)


def test_a_kept_forecast_is_scored_once_its_week_is_over(db):
    add_goalie_game(db, 1, S, 1, date="2026-11-17")
    add_goalie_game(db, 1, S, 2, date="2026-11-19")
    add_goalie_game(db, 1, S, 3, date="2026-11-25")  # the next week: a log after the end
    row = goalie_board.GoalieRow(1, "One", "AAA", 0.75, 1.5, 20.0, None, "", "free agent")
    goalie_board.save(db, [row], "2026-11-16", "2026-11-22")
    got = goalie_board.check(db)
    assert "1 goalie-days" in got and "mean +0.50" in got
    assert "No finished week" in goalie_board.check(db, before="2026-11-20")
