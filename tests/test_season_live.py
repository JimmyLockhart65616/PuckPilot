"""The live score of a week: Yahoo's own totals for both sides, mid-week.

It arrives inside the matchups response the week calendar was always read from,
and was thrown away. These payloads follow Yahoo's documented matchup shape; the
first real in-season fetch is logged raw so the shape can be checked against it.
"""

from __future__ import annotations

import json

from puckpilot.season.fetch import save_live
from puckpilot.season.matchups import parse_live, parse_matchups

US, THEM = "999.l.1.t.5", "999.l.1.t.11"


def _team(key, name, stats, remaining=None):
    tail = {
        "team_stats": {
            "coverage_type": "week",
            "week": "1",
            "stats": [{"stat": {"stat_id": str(k), "value": v}} for k, v in stats.items()],
        }
    }
    if remaining is not None:
        tail["team_remaining_games"] = {
            "coverage_type": "week",
            "week": 1,
            "total": {"remaining_games": remaining, "live_games": 1, "completed_games": 9},
        }
    return {"team": [[{"team_key": key}, {"name": name}], tail]}


def live_payload(ours, theirs, winners=(), status="midevent", week=1):
    m = {
        "week": str(week),
        "week_start": "2026-09-29",
        "week_end": "2026-10-04",
        "status": status,
        "is_playoffs": "0",
        "stat_winners": [{"stat_winner": w} for w in winners],
        "0": {"teams": {"count": 2, "0": ours, "1": theirs}},
    }
    return {
        "fantasy_content": {
            "team": [[{"team_key": US}], {"matchups": {"count": 1, "0": {"matchup": m}}}]
        }
    }


def test_both_sides_totals_are_read_including_display_only_stats():
    p = live_payload(
        _team(US, "Us", {1: "4", 24: "88", 26: ".915"}, remaining=21),
        _team(THEM, "Them", {1: "6", 24: "71", 26: ".901"}, remaining=26),
    )
    live = parse_live(p, US, week=1)
    assert live.status == "midevent" and live.started
    assert live.ours.stats == {1: 4.0, 24: 88.0, 26: 0.915}
    assert live.theirs.stats[1] == 6.0
    # SA is not scored but is SV%'s denominator, so it is kept.
    assert live.banked({1: "G", 24: "SA", 26: "SV%"}) == {"G": 4.0, "SA": 88.0, "SV%": 0.915}


def test_games_left_are_read_for_both_sides():
    """The number the whole request was about: how many each side has left."""
    live = parse_live(
        live_payload(_team(US, "Us", {}, remaining=21), _team(THEM, "Them", {}, remaining=26)),
        US,
    )
    assert (live.ours.remaining_games, live.theirs.remaining_games) == (21, 26)
    assert live.ours.live_games == 1 and live.ours.completed_games == 9


def test_an_undefined_stat_is_none_not_zero():
    """SV% with no shots faced is "-": a zero would read as a disastrous week."""
    live = parse_live(
        live_payload(_team(US, "Us", {26: "-", 1: ""}), _team(THEM, "Them", {26: ".920"})), US
    )
    assert live.ours.stats == {26: None, 1: None}
    assert live.banked({26: "SV%", 1: "G"}) == {}


def test_who_leads_each_category_is_read_from_our_side():
    live = parse_live(
        live_payload(
            _team(US, "Us", {}),
            _team(THEM, "Them", {}),
            winners=(
                {"stat_id": "1", "winner_team_key": US},
                {"stat_id": "2", "winner_team_key": THEM},
                {"stat_id": "5", "is_tied": "1"},
            ),
        ),
        US,
    )
    assert live.winners == {1: "ours", 2: "theirs", 5: "tie"}


def test_our_side_is_found_whichever_order_yahoo_lists_the_teams():
    live = parse_live(live_payload(_team(THEM, "Them", {1: "6"}), _team(US, "Us", {1: "4"})), US)
    assert live.ours.team_key == US and live.ours.stats[1] == 4.0


def test_a_week_that_is_not_in_the_response_is_none():
    p = live_payload(_team(US, "Us", {}), _team(THEM, "Them", {}), week=1)
    assert parse_live(p, US, week=2) is None
    assert parse_live({"nothing": 1}, US) is None


def test_the_calendar_reader_is_untouched_by_the_stats_blocks():
    """Same response, two readers: the week calendar must parse as before."""
    p = live_payload(_team(US, "Us", {1: "4"}), _team(THEM, "Them", {1: "6"}))
    ms = parse_matchups(p, our_team_key=US)
    assert ms[0].opponent_key == THEM and ms[0].start == "2026-09-29"


def test_every_reading_is_logged_with_its_raw_payload(db):
    """Intra-week state cannot be fetched after the fact; this log is the only
    record of how a week unfolded."""
    p = live_payload(_team(US, "Us", {1: "4"}, remaining=21), _team(THEM, "Them", {1: "6"}))
    save_live(db, "jimmy", "999.l.1", parse_live(p, US), raw=p)
    row = db.execute("SELECT * FROM matchup_snapshots").fetchone()
    assert row["week"] == 1 and row["status"] == "midevent" and row["opponent_key"] == THEM
    assert json.loads(row["ours_json"])["stats"] == {"1": 4.0}
    assert json.loads(row["ours_json"])["remaining_games"] == 21
    assert json.loads(row["raw_json"]) == p
