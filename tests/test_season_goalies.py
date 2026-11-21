"""The starting-goalie ladder.

Measured accuracy of `TrailingStartShareSource` on real seasons, same protocol
in all three (60 sampled dates, first 21 days skipped so history exists):

    2025-26   61.5%   Brier 0.2174
    2024-25   64.8%   Brier 0.2061
    2023-24   62.4%   Brier 0.2103

That is the floor, and it is roughly 30 points below the 90% "noisy
announcement" the bench-regret replay assumed. These tests pin the model's
behaviour, not that number; `test_local_data.py` re-measures it against the
real database.
"""

from __future__ import annotations

from puckpilot.data import store
from puckpilot.season.goalies import (
    ChainedGoalieSource,
    StaticGoalieSource,
    TrailingStartShareSource,
)
from tests.conftest import add_goalie_game, add_player

SEASON = "20262027"


def sched(conn, gid, date, home, away, season=SEASON):
    store.upsert_schedule_game(
        conn,
        game_id=gid,
        season=season,
        game_type=2,
        game_date=date,
        start_time_utc=None,
        home_team=home,
        away_team=away,
    )


def started(conn, pid, gid, date, team, season=SEASON):
    add_goalie_game(conn, pid, season, gid, date=date, started=1)
    conn.execute(
        "UPDATE nhl_game_logs SET team_abbrev = ? WHERE player_id = ? AND game_id = ?",
        (team, pid, gid),
    )


def test_a_team_that_does_not_play_has_no_starter(db):
    add_player(db, 1, "Starter", "G")
    sched(db, 100, "2026-10-05", "TOR", "MTL")
    started(db, 1, 100, "2026-10-05", "TOR")
    db.commit()
    src = TrailingStartShareSource(db, SEASON)
    assert src.starts("2026-10-09") == {}


def test_a_lone_starter_is_near_certain(db):
    add_player(db, 1, "Workhorse", "G")
    for i, d in enumerate(["2026-10-01", "2026-10-03", "2026-10-05"]):
        sched(db, 100 + i, d, "TOR", "MTL")
        started(db, 1, 100 + i, d, "TOR")
    sched(db, 200, "2026-10-08", "TOR", "OTT")
    db.commit()
    p = TrailingStartShareSource(db, SEASON).starts("2026-10-08")
    assert p[1] == 1.0


def test_the_previous_starter_is_demoted_not_excluded(db):
    """Tandems do not strictly alternate - starters ride - so the demotion is a
    weight, not a veto. Strict alternation measured 53% against this model's 61%."""
    add_player(db, 1, "A", "G")
    add_player(db, 2, "B", "G")
    days = ["2026-10-01", "2026-10-03", "2026-10-05", "2026-10-07"]
    for i, d in enumerate(days):
        sched(db, 100 + i, d, "TOR", "MTL")
        started(db, 1 if i % 2 == 0 else 2, 100 + i, d, "TOR")
    sched(db, 300, "2026-10-09", "TOR", "OTT")
    db.commit()
    p = TrailingStartShareSource(db, SEASON).starts("2026-10-09")
    # 2 started the previous game, so 1 is favoured - but 2 keeps real weight
    assert p[1] > p[2] > 0.0


def test_a_backup_with_one_start_in_ten_falls_below_the_floor(db):
    add_player(db, 1, "Starter", "G")
    add_player(db, 2, "Backup", "G")
    days = [f"2026-10-{d:02d}" for d in range(1, 22, 2)]
    for i, d in enumerate(days):
        sched(db, 100 + i, d, "TOR", "MTL")
        started(db, 2 if i == 0 else 1, 100 + i, d, "TOR")
    sched(db, 400, "2026-10-25", "TOR", "OTT")
    db.commit()
    p = TrailingStartShareSource(db, SEASON).starts("2026-10-25")
    assert p.get(1, 0) > 0.8
    assert 2 not in p  # one start in the window is not a lineup decision


def test_only_games_before_the_date_are_used(db):
    """No lookahead: a start tonight must not inform tonight's prediction."""
    add_player(db, 1, "A", "G")
    add_player(db, 2, "B", "G")
    sched(db, 100, "2026-10-05", "TOR", "MTL")
    started(db, 1, 100, "2026-10-05", "TOR")
    sched(db, 101, "2026-10-07", "TOR", "OTT")
    started(db, 2, 101, "2026-10-07", "TOR")  # tonight's actual starter
    db.commit()
    p = TrailingStartShareSource(db, SEASON).starts("2026-10-07")
    assert p == {1: 1.0}


def test_last_season_carries_the_opening_weeks(db):
    """In October the current season has no history; last season's share is a
    better prior than nothing."""
    add_player(db, 1, "Last Year Starter", "G")
    sched(db, 50, "2026-04-01", "TOR", "MTL", season="20252026")
    started(db, 1, 50, "2026-04-01", "TOR", season="20252026")
    sched(db, 100, "2026-09-29", "TOR", "OTT")
    db.commit()
    assert TrailingStartShareSource(db, SEASON).starts("2026-09-29") == {}
    with_fb = TrailingStartShareSource(db, SEASON, fallback_season="20252026")
    assert with_fb.starts("2026-09-29") == {1: 1.0}


def test_relief_appearances_are_not_starts(db):
    add_player(db, 1, "Reliever", "G")
    sched(db, 100, "2026-10-05", "TOR", "MTL")
    add_goalie_game(db, 1, SEASON, 100, date="2026-10-05", started=0)
    sched(db, 101, "2026-10-07", "TOR", "OTT")
    db.commit()
    assert TrailingStartShareSource(db, SEASON).starts("2026-10-07") == {}


# -- the chain --------------------------------------------------------------


def test_an_earlier_source_overrides_a_later_one():
    announced = StaticGoalieSource({"2026-10-07": {1: 1.0}})
    model = StaticGoalieSource({"2026-10-07": {1: 0.55, 2: 0.45}})
    chained = ChainedGoalieSource(announced, model)
    p = chained.starts("2026-10-07")
    assert p[1] == 1.0  # the announcement wins
    assert p[2] == 0.45  # the model still answers for everyone else


def test_a_source_that_knows_nothing_today_costs_nothing():
    chained = ChainedGoalieSource(StaticGoalieSource({}), StaticGoalieSource({"d": {1: 0.6}}))
    assert chained.starts("d") == {1: 0.6}


def test_a_dead_source_degrades_to_the_floor_not_to_an_empty_lineup():
    class Broken:
        def starts(self, date):
            raise RuntimeError("feed is down")

    chained = ChainedGoalieSource(Broken(), StaticGoalieSource({"d": {1: 0.6}}))
    assert chained.starts("d") == {1: 0.6}


def test_a_frozen_source_does_not_see_games_after_its_cutoff(db):
    """Replay only: asked on Wednesday about Saturday, the trailing model would
    count Thursday's start. Frozen at Wednesday, it cannot."""
    from puckpilot.season.goalies import AsOfGoalieSource

    add_player(db, 1, "A", "G")
    add_player(db, 2, "B", "G")
    sched(db, 100, "2026-10-05", "TOR", "MTL")
    started(db, 1, 100, "2026-10-05", "TOR")
    sched(db, 101, "2026-10-08", "TOR", "OTT")
    started(db, 2, 101, "2026-10-08", "TOR")  # Thursday: after the cutoff
    sched(db, 102, "2026-10-10", "TOR", "BOS")
    db.commit()
    src = TrailingStartShareSource(db, SEASON)
    assert src.starts("2026-10-10") != {1: 1.0}  # unfrozen: Thursday counts
    frozen = AsOfGoalieSource(src, "2026-10-07")
    assert frozen.starts("2026-10-10") == {1: 1.0}
    assert frozen.starts("2026-10-09") == {}  # TOR do not play that day


# -- trades, absences and the games after the next ---------------------------


def dressed(conn, gid, team, *pids, season=SEASON):
    """Boxscore rows for the goalies who dressed - starter or not."""
    import json

    store.upsert_boxscore_rows(
        conn,
        [(gid, pid, season, team, json.dumps({"playerId": pid, "position": "G"})) for pid in pids],
    )


def game(conn, gid, date, team, starter, *bench, opp="ZZZ"):
    sched(conn, gid, date, team, opp)
    started(conn, starter, gid, date, team)
    dressed(conn, gid, team, starter, *bench)


def test_a_traded_goalie_counts_only_for_his_new_club(db):
    """Starts are keyed by the club he made them for, so his old club's window
    still held him - and on a date both clubs played he got whichever club's
    number came last, which changed with the interpreter's hash seed."""
    for pid in (1, 2, 3):
        add_player(db, pid, f"G{pid}", "G")
    for i, (d, who) in enumerate(
        [("2026-10-01", 1), ("2026-10-03", 1), ("2026-10-05", 2), ("2026-10-07", 1)]
    ):
        game(db, 100 + i, d, "TOR", who, 2 if who == 1 else 1)
    for i, d in enumerate(["2026-10-02", "2026-10-04", "2026-10-06"]):
        game(db, 200 + i, d, "MTL", 3, 4, opp="YYY")
    game(db, 210, "2026-10-08", "MTL", 1, 3, opp="YYY")  # after the trade
    sched(db, 300, "2026-10-10", "TOR", "OTT")
    sched(db, 301, "2026-10-10", "MTL", "BOS")
    db.commit()

    src = TrailingStartShareSource(db, SEASON, current_club=True)
    assert src.team_starts("TOR", "2026-10-10") == {2: 1.0}
    p = src.starts("2026-10-10")
    assert p[1] == src.team_starts("MTL", "2026-10-10")[1] < 0.5
    assert p[2] == 1.0


def test_a_starter_who_stops_dressing_hands_his_share_to_whoever_is_left(db):
    """Two games without dressing is an injury, a demotion or a trade. Before,
    his old share kept his partner below the 0.5 floor every night of it."""
    for pid in (1, 2, 3):
        add_player(db, pid, f"G{pid}", "G")
    days = [f"2026-10-{d:02d}" for d in (1, 3, 5, 7, 9, 11, 13)]
    for i, d in enumerate(days[:5]):
        game(db, 100 + i, d, "TOR", 2 if i == 2 else 1, 1 if i == 2 else 2)
    for i, d in enumerate(days[5:]):
        game(db, 110 + i, d, "TOR", 2, 3)  # 1 is hurt: a call-up dresses
    sched(db, 120, "2026-10-15", "TOR", "OTT")
    db.commit()

    # The window alone (no prior, no back-to-back rule): what the dressed rule changes.
    plain = {"prior_games": 0.0, "b2b_same": None}
    before = TrailingStartShareSource(db, SEASON, dressed_window=None, **plain)
    assert before.starts("2026-10-15")[2] < 0.5  # the old model: the healthy goalie under the floor
    after = TrailingStartShareSource(db, SEASON, dressed_window=2, **plain).starts("2026-10-15")
    assert after == {2: 1.0}
    assert TrailingStartShareSource(db, SEASON, **plain).starts("2026-10-15") == after  # default


def _tandem(db):
    """Ten games, 1 started six and 2 four, and 1 started the last."""
    add_player(db, 1, "A", "G")
    add_player(db, 2, "B", "G")
    who = [2, 1, 2, 1, 2, 1, 2, 1, 1, 1]
    for i, g in enumerate(who):
        game(db, 100 + i, f"2026-10-{i + 1:02d}", "TOR", g, 3 - g)
    sched(db, 200, "2026-10-12", "TOR", "OTT")
    sched(db, 201, "2026-10-14", "TOR", "BOS")
    db.commit()


def test_the_previous_starter_is_demoted_for_the_next_game_only(db):
    _tandem(db)
    old = TrailingStartShareSource(db, SEASON, ahead="damped")
    share = TrailingStartShareSource(db, SEASON, ahead="share")
    chain = TrailingStartShareSource(db, SEASON)  # the default
    tonight = old.starts("2026-10-12")
    assert tonight == {1: 0.4286, 2: 0.5714}  # 6 x 0.5 against 4
    # Tonight is the same under every rule: only later games differ.
    assert share.starts("2026-10-12") == chain.starts("2026-10-12") == tonight
    # The old rule demoted 1 for the game after as well.
    assert old.starts("2026-10-14") == tonight
    assert share.starts("2026-10-14") == {1: 0.6, 2: 0.4}
    # The chain: 1 is likelier to start again after a night off.
    later = chain.starts("2026-10-14")
    assert later[1] > 0.5 > later[2]
    for p in (tonight, share.starts("2026-10-14"), later):
        assert abs(sum(p.values()) - 1.0) < 1e-3


def test_a_frozen_source_counts_the_games_between_cutoff_and_date(db):
    """Asked on the 12th about the 14th, the 14th is the second game ahead."""
    from puckpilot.season.goalies import AsOfGoalieSource

    _tandem(db)
    chain = TrailingStartShareSource(db, SEASON, ahead="chain")
    frozen = AsOfGoalieSource(chain, "2026-10-12")
    assert frozen.starts("2026-10-12") == chain.starts("2026-10-12")
    assert frozen.starts("2026-10-14") == chain.starts("2026-10-14")


def test_a_goalie_known_to_be_out_is_left_out_today(db):
    _tandem(db)
    src = TrailingStartShareSource(db, SEASON, unavailable={1})
    assert src.starts("2026-10-12") == {2: 1.0}


def test_model_specs_read_from_the_command_line():
    import pytest

    from puckpilot.season.goalies import parse_spec

    assert parse_spec("") == {}
    assert parse_spec("old") == {
        "current_club": False,
        "dressed_window": None,
        "ahead": "damped",
        "prior_games": 0.0,
        "b2b_same": None,
    }
    assert parse_spec("club-dw2-chain-p0-boff")["b2b_same"] is None
    assert parse_spec("club-dw2-chain") == {
        "current_club": True,
        "dressed_window": 2,
        "ahead": "chain",
    }
    assert parse_spec("t12-x0.4") == {"trailing_games": 12, "previous_damping": 0.4}
    with pytest.raises(ValueError):
        parse_spec("club-sideways")
    with pytest.raises(ValueError):
        parse_spec("xfast")


# -- the rest of the season ----------------------------------------------------


def _ten_games(db):
    """TOR's ten games: 1 started seven, 2 started three; both dressed every one."""
    add_player(db, 1, "Starter", "G")
    add_player(db, 2, "Backup", "G")
    for i in range(10):
        g = 2 if i in (2, 5, 8) else 1
        game(db, 100 + i, f"2026-10-{i + 1:02d}", "TOR", g, 3 - g)
    db.commit()


def test_a_share_so_far_is_shrunk_toward_the_projection(db):
    from puckpilot.season.goalies import GoalieWorkload

    _ten_games(db)
    plain = GoalieWorkload(db, SEASON)
    assert plain.share(1, "2026-10-11") == 0.7
    assert plain.share(2, "2026-10-11") == 0.3
    w = GoalieWorkload(db, SEASON, priors={1: 0.6, 2: 0.4}, prior_games=10)
    assert w.share(1, "2026-10-11") == (7 + 6) / 20
    assert w.share(2, "2026-10-11") == (3 + 4) / 20
    # Nothing played yet: the projection alone; neither: no answer.
    assert w.share(1, "2026-10-01") == 0.6
    assert GoalieWorkload(db, SEASON).share(9, "2026-10-11") is None
    assert w.shares("2026-10-11", [1, 2, 9]) == {1: 0.65, 2: 0.35}


def test_a_traded_goalie_is_judged_on_his_new_club_since_he_joined(db):
    from puckpilot.season.goalies import GoalieWorkload

    _ten_games(db)
    game(db, 300, "2026-10-12", "MTL", 3, 1, opp="YYY")  # 1 dresses for MTL
    game(db, 301, "2026-10-14", "MTL", 1, 3, opp="YYY")
    db.commit()
    assert GoalieWorkload(db, SEASON).share(1, "2026-10-15") == 0.5


def test_a_backup_is_worth_his_share_of_the_season_not_every_game():
    """Counting every club game made a backup look two to three times what he
    is, and an upgrade or a keeper on any measure taken over the season."""
    from types import SimpleNamespace

    from puckpilot.season.add_story import SeasonLeft
    from puckpilot.season.week import _season_value

    def p(key, pid, pos):
        return SimpleNamespace(player_key=key, nhl_player_id=pid, team="TOR", position=pos)

    values = SimpleNamespace(per_game=lambda pid, day: 1.0)
    roster = [p("g", 1, "G"), p("s", 2, "C")]
    flat = _season_value(roster, values, {"TOR": 40}, "2026-10-12")
    assert flat == {"g": 40.0, "s": 40.0}  # without workload, as before
    shared = _season_value(roster, values, SeasonLeft({"TOR": 40}, {1: 0.3}), "2026-10-12")
    assert shared == {"g": 12.0, "s": 40.0}


def test_projected_shares_come_from_projected_games():
    import pandas as pd

    from puckpilot.season.goalies import projected_shares

    frame = pd.DataFrame(
        {"proj_gp": [55.0, 20.0, 70.0, float("nan")], "position": ["G", "G", "C", "G"]},
        index=[1, 2, 3, 4],
    )
    assert projected_shares(frame, 82) == {1: 55 / 82, 2: 20 / 82}
    assert projected_shares(None, 82) == {}


# -- the gate ---------------------------------------------------------------------


def test_the_gate_scores_a_certain_starter_as_perfect(db, monkeypatch):
    from puckpilot.season import goalie_gate

    monkeypatch.setattr(goalie_gate, "SKIP_DATES", 3)
    add_player(db, 1, "Workhorse", "G")
    add_player(db, 2, "Backup", "G")
    for i in range(8):
        game(db, 100 + i, f"2026-10-{i + 1:02d}", "TOR", 1, 2)
    db.commit()
    # Eight straight days are all back-to-backs; the scoring is the subject here,
    # so the model is the one without the back-to-back rule or a prior.
    s = goalie_gate.score_variant(db, SEASON, "club-dw2-chain-p0-boff", None)
    assert s.n == 5
    assert s.mean("hit") == 1.0
    assert s.mean("brier") == 0.0
    assert s.week_error() == 0.0
    assert s.ahead_brier() == 0.0


def test_the_gate_brier_counts_the_starter_the_model_left_out():
    from puckpilot.season.goalie_gate import _brier, _clustered, _favourite

    assert abs(_brier({1: 0.7, 2: 0.3}, 1) - 0.18) < 1e-12
    assert _brier({1: 1.0}, 2) == 2.0  # named the wrong one, missed the right one
    assert _brier({}, 2) == 1.0
    assert _favourite({1: 0.5, 2: 0.5}) == 1  # ties go to the lower id, every time
    mean, se = _clustered([("A", 1.0), ("A", 1.0), ("B", -1.0), ("B", -1.0)])
    assert mean == 0.0 and se > 0


def test_a_goalie_tagged_healthy_is_not_written_off_for_missing_two_games(db):
    """The night a starter returns from injury he has not dressed for two
    games - exactly what the rule reads as out. A clean tag says otherwise."""
    for pid in (1, 2, 3):
        add_player(db, pid, f"G{pid}", "G")
    days = [f"2026-10-{d:02d}" for d in (1, 3, 5, 7, 9, 11, 13)]
    for i, d in enumerate(days[:5]):
        game(db, 100 + i, d, "TOR", 1, 2)
    for i, d in enumerate(days[5:]):
        game(db, 110 + i, d, "TOR", 2, 3)
    sched(db, 120, "2026-10-15", "TOR", "OTT")
    db.commit()
    gone = TrailingStartShareSource(db, SEASON, dressed_window=2).starts("2026-10-15")
    back = TrailingStartShareSource(db, SEASON, dressed_window=2, healthy={1}).starts("2026-10-15")
    assert 1 not in gone
    assert back[1] > back[2]


# -- the opening weeks and back-to-backs (2026-10-11) -------------------------

PRIOR = "20252026"


def _sole_starter(db, days, next_days):
    """TOR: goalie 1 started every game so far, 2 dressed as his backup."""
    add_player(db, 1, "Starter", "G")
    add_player(db, 2, "Backup", "G")
    for i, d in enumerate(days):
        game(db, 100 + i, d, "TOR", 1, 2)
    for i, d in enumerate(next_days):
        sched(db, 200 + i, d, "TOR", "OTT")
    db.commit()


def test_a_sole_starter_is_certain_on_a_back_to_back_without_the_rule(db):
    """What the live model said of a sole starter for the second night of a
    back-to-back he did not start (2026-10-10)."""
    _sole_starter(db, ["2026-10-01", "2026-10-03", "2026-10-05"], ["2026-10-06"])
    old = TrailingStartShareSource(db, SEASON, prior_games=0, b2b_same=None)
    assert old.starts("2026-10-06") == {1: 1.0}


def test_the_second_night_of_a_back_to_back_goes_mostly_to_the_backup(db):
    """Measured 4.5-8.1%: the previous night's starter rarely starts again.
    His backup has no start of his own, so the demotion alone could not move
    anything to him."""
    _sole_starter(db, ["2026-10-01", "2026-10-03", "2026-10-05"], ["2026-10-06"])
    src = TrailingStartShareSource(db, SEASON, prior_games=0, b2b_same=0.07)
    assert src.starts("2026-10-06") == {1: 0.07, 2: 0.93}


def test_a_back_to_back_later_in_the_week_is_walked_through_the_chain(db):
    _sole_starter(db, ["2026-10-01", "2026-10-03", "2026-10-05"], ["2026-10-08", "2026-10-09"])
    src = TrailingStartShareSource(db, SEASON, prior_games=0, b2b_same=0.07)
    assert src.starts("2026-10-08") == {1: 1.0}  # rested: no back-to-back
    assert src.starts("2026-10-09") == {1: 0.07, 2: 0.93}
    plain = TrailingStartShareSource(db, SEASON, prior_games=0, b2b_same=None)
    assert plain.starts("2026-10-09") == {1: 1.0}


def _with_last_season(db):
    """Last season TOR's 1 started three of four, 2 one; 3 started for TOR then
    but has not dressed for it this season."""
    for pid in (1, 2, 3):
        add_player(db, pid, f"G{pid}", "G")
    for i, (d, who) in enumerate(
        [("2026-03-01", 1), ("2026-03-03", 1), ("2026-03-05", 2), ("2026-03-07", 3)]
    ):
        sched(db, 900 + i, d, "TOR", "MTL", season=PRIOR)
        started(db, who, 900 + i, d, "TOR", season=PRIOR)


def test_the_prior_pads_a_short_window_with_last_seasons_shares(db):
    """Two games in, the window is two starts by one goalie. Padded to ten with
    last season's shares - 1: 2/4, 2: 1/4 of a season, normalised over the
    goalies who have dressed for TOR (3 has not, so he keeps nothing)."""
    _with_last_season(db)
    game(db, 100, "2026-10-01", "TOR", 1, 2)
    game(db, 101, "2026-10-03", "TOR", 1, 2)
    sched(db, 200, "2026-10-05", "TOR", "OTT")
    db.commit()
    plain = TrailingStartShareSource(db, SEASON, fallback_season=PRIOR, prior_games=0)
    assert plain.starts("2026-10-05") == {1: 1.0}
    src = TrailingStartShareSource(db, SEASON, fallback_season=PRIOR, prior_games=10)
    # 1: 2 + 8 x 2/3, demoted by half as last night's starter; 2: 8 x 1/3.
    w1, w2 = (2 + 8 * 2 / 3) * 0.5, 8 / 3
    p = src.starts("2026-10-05")
    assert p == {1: round(w1 / (w1 + w2), 4), 2: round(w2 / (w1 + w2), 4)}
    assert 3 not in p


def test_the_prior_fades_out_once_the_season_fills_the_window(db):
    _with_last_season(db)
    for i in range(10):
        game(db, 100 + i, f"2026-10-{i + 1:02d}", "TOR", 1, 2)
    sched(db, 200, "2026-10-12", "TOR", "OTT")
    db.commit()
    plain = TrailingStartShareSource(db, SEASON, fallback_season=PRIOR, prior_games=0)
    padded = TrailingStartShareSource(db, SEASON, fallback_season=PRIOR, prior_games=10)
    assert padded.starts("2026-10-12") == plain.starts("2026-10-12") == {1: 1.0}


def test_a_goalie_with_no_starts_last_season_gets_the_unknown_share(db):
    from puckpilot.season.goalies import UNKNOWN_PRIOR_SHARE

    _with_last_season(db)
    add_player(db, 4, "Rookie", "G")
    game(db, 100, "2026-10-01", "TOR", 4, 1)
    sched(db, 200, "2026-10-03", "TOR", "OTT")
    db.commit()
    src = TrailingStartShareSource(db, SEASON, fallback_season=PRIOR, prior_games=10)
    shares = {1: 0.5, 4: UNKNOWN_PRIOR_SHARE}
    total = sum(shares.values())
    w1 = 9 * shares[1] / total
    w4 = (1 + 9 * shares[4] / total) * 0.5  # he started last night
    assert src.starts("2026-10-03") == {1: round(w1 / (w1 + w4), 4), 4: round(w4 / (w1 + w4), 4)}


def test_the_new_knobs_read_from_a_spec_and_are_checked():
    import pytest

    from puckpilot.season.goalies import parse_spec

    assert parse_spec("club-dw2-chain-p10-b0.07") == {
        "current_club": True,
        "dressed_window": 2,
        "ahead": "chain",
        "prior_games": 10.0,
        "b2b_same": 0.07,
    }
    with pytest.raises(ValueError):
        parse_spec("bsoon")


def test_impossible_knob_values_are_refused(db):
    import pytest

    with pytest.raises(ValueError):
        TrailingStartShareSource(db, SEASON, b2b_same=1.5)
    with pytest.raises(ValueError):
        TrailingStartShareSource(db, SEASON, prior_games=-1)


def test_the_gate_scores_the_opening_dates_apart():
    import pytest

    from puckpilot.season.goalie_gate import SKIP_DATES, _eval_dates

    class _Src:
        _starts = {(f"2026-10-{d:02d}", "TOR"): 1 for d in range(1, 31)}

    dates = sorted({d for d, _ in _Src._starts})
    assert _eval_dates(_Src) == dates[SKIP_DATES:]
    assert _eval_dates(_Src, "early") == dates[:SKIP_DATES]
    assert _eval_dates(_Src, "all") == dates
    with pytest.raises(ValueError):
        _eval_dates(_Src, "midseason")


def test_the_default_model_is_the_measured_one():
    """2026-10-11: the prior and the back-to-back rule passed every
    pre-registered check (goalie-check early and late, add gate -g)."""
    from puckpilot.season import goalies

    assert goalies.PRIOR_GAMES == 10.0 and goalies.B2B_SAME == 0.07
