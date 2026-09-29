"""An add's reasons, in the terms a person decides it by."""

from __future__ import annotations

from types import SimpleNamespace

from puckpilot.engine.categories import resolve
from puckpilot.season import add_story
from puckpilot.season.odds import CategoryOdds, OddsModel, WeekOdds


def _p(key, name, pos="C", team="TOR", pid=1):
    return SimpleNamespace(
        player_key=key, name=name, position=pos, team=team, nhl_player_id=pid, on_ir=False
    )


def _odds(**chances):
    return WeekOdds(
        tuple(CategoryOdds(resolve(label), p, 0.0, 0.0, 0.0) for label, p in chances.items())
    )


def test_every_category_the_swap_moves_is_listed_down_as_well_as_up():
    """Dropping a goalie for a skater wins shots and costs saves; hiding the
    cost is how a card talks someone into a bad add."""
    got = add_story.odds_lines(_odds(SOG=0.30, SV=0.90, G=0.50), _odds(SOG=0.48, SV=0.75, G=0.505))
    assert got == ["SOG 30% -> 48%", "SV 90% -> 75%"]


def test_a_swap_that_moves_nothing_says_so():
    assert add_story.odds_lines(_odds(G=0.5), _odds(G=0.5)) == [
        "No category's chance moves by a point."
    ]


def test_a_goalie_range_counts_only_the_starts_he_may_get():
    """Two games at 60% each is not two starts: the floor is none at all."""
    g = _p("g.1", "Backup", pos="G")
    lineups = {
        "2026-10-05": ({}, {"g.1": 0.6}, {"TOR"}),
        "2026-10-06": ({}, {"g.1": 0.6}, {"TOR"}),
    }
    got = add_story.range_line(g, lineups, {"wins": 0.5, "saves": 26.0}, OddsModel())
    assert got.startswith("Backup, 1.2 expected starts: W 0-")
    assert "SV 0-" in got


def test_a_skater_range_is_his_lineup_starts_not_his_team_games():
    s = _p("s.1", "Grinder")
    lineups = {
        "2026-10-05": ({"s.1": "C"}, {}, {"TOR"}),
        "2026-10-06": ({}, {}, {"TOR"}),  # plays, but the lineup is full
    }
    got = add_story.range_line(s, lineups, {"sog": 3.0, "hits": 2.0}, OddsModel(p_play=1.0))
    assert got.startswith("Grinder, 1 start: ")
    assert "SOG" in got and "HIT" in got


def test_a_player_with_no_starts_left_has_no_range():
    got = add_story.range_line(_p("s.1", "Idle"), {"2026-10-05": ({}, {}, set())}, {}, OddsModel())
    assert got == "Idle: no starts left this week"


def test_taking_the_drops_slot_is_said_as_such():
    cand, drop = _p("a", "New"), _p("d", "Old")
    before = {"2026-10-05": ({"d": "C"}, {}, {"TOR"})}
    after = {"2026-10-05": ({"a": "C"}, {}, {"TOR"})}
    got = add_story.week_lines(before, after, [drop], [cand], cand, drop)
    assert got == [
        "New: 1 game left this week, 1 in your lineup",
        "Mon 5: New starts in Old's place (C)",
    ]


def test_a_drop_that_loses_starts_the_add_cannot_use_is_named():
    cand, drop = _p("a", "New", team="MTL"), _p("d", "Old")
    before = {"2026-10-05": ({"d": "C"}, {}, {"TOR"})}
    after = {"2026-10-05": ({}, {}, {"TOR"})}
    got = add_story.week_lines(before, after, [drop], [cand], cand, drop)
    assert got[-1] == "Mon 5: Old loses a start at C"


def test_the_season_view_ranks_both_on_the_roster():
    values = SimpleNamespace(per_game=lambda pid, day: {1: 2.0, 2: 1.0, 3: 0.5, 9: 1.5}[pid])
    roster = [_p("r1", "Top", pid=1), _p("r2", "Mid", pid=2), _p("r3", "Low", pid=3)]
    got = add_story.season_lines(
        _p("fa", "Pickup", pid=9), roster[2], roster, values, {"TOR": 10}, "2026-10-05"
    )
    assert got[0] == "Pickup would rank 2 of 3 on your roster"
    assert got[1] == "Low ranks 3 of 3 - your lowest"
    assert "worth more" in got[2]
