"""In-week goalie adds as a stream of expected starts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from puckpilot.season import streams
from tests.conftest import add_goalie_game

S = "20262027"


def test_starts_still_to_come_scale_with_the_days_left_squared():
    g = streams.GoalieStream(starts=2.0, shots=28.0, save_pct=0.9, p_win=0.5, team_weeks=10)
    assert g.expected(7, 7) == 2.0
    assert g.expected(0, 7) == 0.0
    assert g.expected(7 // 2, 7) == pytest.approx(2.0 * (3 / 7) ** 2)
    chances = g.chances(7, 7)
    assert chances == [1.0, 1.0] and all(c <= 1 for c in g.chances(5, 7))
    assert sum(g.chances(5, 7)) == pytest.approx(g.expected(5, 7))
    r = g.rates()
    assert r["saves"] == pytest.approx(25.2) and r["shots_against"] == 28.0
    assert r["wins"] == 0.5


def _add(db, team, pid, at):
    db.execute(
        "INSERT INTO league_transactions (league_key, txn_key, at, type, team_key, action,"
        " player_key, nhl_player_id, positions)"
        " VALUES ('999.l.1', ?, ?, 'add', ?, 'add', ?, ?, 'G')",
        (f"t{pid}{at}", at, team, f"p{pid}", pid),
    )


def test_a_teams_stream_is_its_own_adds_shrunk_toward_the_league(db):
    week = SimpleNamespace(number=1, start="2026-10-06", end="2026-10-12")
    _add(db, "A", 1, "2026-10-08T15:00:00Z")
    add_goalie_game(db, 1, S, 1, date="2026-10-07", shots_against=40, goals_against=0)  # before
    add_goalie_game(db, 1, S, 2, date="2026-10-09", shots_against=30, goals_against=3, decision="W")
    add_goalie_game(db, 1, S, 3, date="2026-10-11", shots_against=20, goals_against=2)
    add_goalie_game(db, 1, S, 4, date="2026-10-12", started=0, shots_against=5, goals_against=0)
    added = streams.added_by_week(db, "999.l.1", S, [week])
    league = streams.measure(added, [1], ["A", "B"])
    assert league.starts == 1.0  # two starts over two team-weeks
    assert league.shots == 25.0 and league.save_pct == pytest.approx(0.9)
    assert league.p_win == 0.5
    a = streams.measure(added, [1], ["A", "B"], team="A", shrink_weeks=1.0)
    b = streams.measure(added, [1], ["A", "B"], team="B", shrink_weeks=1.0)
    assert (a.starts, b.starts) == (1.5, 0.5)
    assert streams.measure(added, [], ["A", "B"]) is None


def test_the_opponents_stream_enters_their_projected_saves_and_ours_is_refused_to_the_search(db):
    import pandas as pd

    from puckpilot.engine.categories import resolve
    from puckpilot.league import LeagueConfig
    from puckpilot.season.odds import OddsModel
    from puckpilot.season.roster import TeamRoster
    from puckpilot.season.week import build_week_plan
    from tests.test_season_settings import _slots
    from tests.test_season_week import _PerGame, _rp, _runtime_for_week

    rt = _runtime_for_week(roster_positions=_slots(("C", 1, 1), ("BN", 1, 0)))
    ours = TeamRoster(
        league_key="l", team_key="a", date="d", players=(_rp("p.1", "A", 1, "TOR", "C"),)
    )
    theirs = TeamRoster(
        league_key="l", team_key="b", date="d", players=(_rp("p.2", "B", 2, "OTT", "C"),)
    )
    frame = pd.DataFrame({"goals": [41.0, 20.5], "proj_gp": [82.0, 82.0]}, index=[1, 2])
    league = LeagueConfig(skater_cats=(resolve("G"),), goalie_cats=(resolve("SV"),))
    stream = streams.GoalieStream(starts=2.0, shots=28.0, save_pct=0.9, p_win=0.5, team_weeks=9)

    def plan(**kw):
        return build_week_plan(
            db, rt, league, rt.week(1), "B", ours, theirs, [], frame, None,
            _PerGame({1: 1.0, 2: 1.0}), find_targets=False, odds_model=OddsModel(), **kw,
        )  # fmt: skip

    def saves(p):
        o = next(o for o in p.outlook if o.category.key == "saves")
        return o.ours, o.theirs

    base = saves(plan())
    with_theirs = saves(plan(their_stream=stream))
    assert with_theirs[0] == base[0]
    assert with_theirs[1] == pytest.approx(base[1] + 2 * 25.2)
    with pytest.raises(ValueError):
        build_week_plan(
            db, rt, league, rt.week(1), "B", ours, theirs, [], frame, None,
            _PerGame({1: 1.0, 2: 1.0}), our_stream=stream,
        )  # fmt: skip


def test_the_their_stream_arm_parses():
    from puckpilot.season.add_gate import parse_arm

    assert parse_arm("odds-daily-ts").their_stream and not parse_arm("odds-daily").their_stream
