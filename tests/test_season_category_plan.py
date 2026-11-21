"""The season's category plan: records, all-play, stances and pins."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from puckpilot.engine.categories import resolve
from puckpilot.league import LeagueConfig
from puckpilot.season import category_plan as cp

LEAGUE = LeagueConfig(skater_cats=(resolve("G"), resolve("HIT")), goalie_cats=(resolve("SV"),))
RUNTIME = SimpleNamespace(league_key="999.l.1", stat_labels=lambda: {1: "G", 31: "HIT", 25: "SV"})


def _week(db, week, team, opp, g, hit, sv):
    db.execute(
        "INSERT INTO league_week_stats (league_key, week, team_key, name, opponent_key, status,"
        " cats_won, completed_games, stats_json, fetched_at)"
        " VALUES ('999.l.1', ?, ?, ?, ?, 'postevent', 0, 0, ?, '')",
        (week, team, team, opp, json.dumps({"1": g, "31": hit, "25": sv})),
    )


def test_a_record_is_against_the_opponent_and_all_play_against_everyone(db):
    _week(db, 1, "a", "b", 5, 40, 100)
    _week(db, 1, "b", "a", 3, 50, 120)
    _week(db, 1, "c", "d", 7, 10, 90)
    _week(db, 1, "d", "c", 5, 60, 150)
    got = cp.records(db, RUNTIME, LEAGUE, "a")
    assert got["goals"] == (1, 0, 0, pytest.approx((1 + 0 + 0.5) / 3))
    assert got["hits"][:3] == (0, 1, 0) and got["hits"][3] == pytest.approx(1 / 3)
    assert got["saves"][:3] == (0, 1, 0) and got["saves"][3] == pytest.approx(1 / 3)


def _line(key, per_spot_lev, share, pinned=None):
    return cp.CategoryLine(
        key=key, label=key, won=0, lost=0, tied=0, all_play=share, leverage=per_spot_lev,
        starter_week=1.0, weeks_left=1, proposed=cp.Stance.HOLD, pinned=pinned,
    )  # fmt: skip


def test_the_proposal_chases_the_best_buys_and_concedes_the_worst_only_when_losing():
    lines = [
        _line("a", 3.0, 0.5),
        _line("b", 2.0, 0.5),
        _line("c", 1.0, 0.2),  # cheap and losing: concede
        _line("d", 0.9, 0.8),  # cheap and winning: hold
        _line("e", 2.5, 0.1),
        _line("f", 1.5, 0.5),
    ]
    got = cp.propose(lines)
    assert got["a"] == got["e"] == cp.Stance.CHASE
    assert got["c"] == cp.Stance.CONCEDE and got["d"] == cp.Stance.HOLD
    assert got["b"] == got["f"] == cp.Stance.HOLD


def test_a_pin_is_kept_shown_and_taken_off(db):
    cp.pin(db, "m", "999.l.1", "assists", cp.Stance.CHASE)
    cp.pin(db, "m", "999.l.1", "ppp", cp.Stance.CONCEDE)
    assert cp.pins(db, "m", "999.l.1") == {"assists": cp.Stance.CHASE, "ppp": cp.Stance.CONCEDE}
    assert cp.tilt(cp.pins(db, "m", "999.l.1")) == {"assists": 1.5, "ppp": 0.0}
    cp.pin(db, "m", "999.l.1", "ppp", None)
    assert cp.pins(db, "m", "999.l.1") == {"assists": cp.Stance.CHASE}
    assert cp.pins(db, "other", "999.l.1") == {}


def test_nothing_pinned_is_no_tilt_and_a_line_shows_its_stance():
    assert cp.tilt({}) == {}
    ln = _line("a", 2.0, 0.5, pinned=cp.Stance.CONCEDE)
    assert ln.stance is cp.Stance.CONCEDE and ln.per_spot == 2.0
    lines = cp.build({}, LEAGUE, {"goals": 0.1}, {"goals": 2.0}, 10, {})
    assert [ln.key for ln in lines] == ["goals", "hits", "saves"]
    assert lines[0].per_spot == pytest.approx(2.0) and lines[1].per_spot is None
    assert "Category plan" in cp.text(lines, 0.0) and "stances do nothing" in cp.text(lines, 0.0)
