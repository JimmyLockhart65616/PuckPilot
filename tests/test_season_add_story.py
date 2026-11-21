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


# -- who they are ------------------------------------------------------------------


def test_a_skater_is_summed_from_his_game_logs_and_boxscores(db):
    from tests.conftest import add_skater_game

    add_skater_game(
        db, 7, "20252026", 1, goals=1, assists=1, points=2, shots=4, toi="18:30", hits=3, blocks=1
    )
    add_skater_game(
        db, 7, "20252026", 2, assists=1, points=1, powerPlayPoints=1, shots=2, toi="17:30", hits=2
    )
    p = SimpleNamespace(nhl_player_id=7, position="C")
    assert add_story.form_line(p, db, "20252026") == (
        "25-26: 2 GP, 1-2-3, 1 PPP, 6 SOG, 5 HIT, 1 BLK, 18.0 min a game"
    )
    assert add_story.form_line(p, db, "20262027") == "26-27: no games"


def test_a_goalie_is_sized_up_by_wins_and_save_percentage(db):
    from tests.conftest import add_skater_game

    add_skater_game(db, 8, "20252026", 1, shotsAgainst=30, goalsAgainst=2, decision="W")
    add_skater_game(db, 8, "20252026", 2, shotsAgainst=20, goalsAgainst=3, decision="L")
    p = SimpleNamespace(nhl_player_id=8, position="G")
    assert add_story.form_line(p, db, "20252026") == (
        "25-26: 2 GP, 1 W, 0.900 SV%, 25.0 shots faced a game"
    )


def test_the_profile_names_both_players_with_age_and_both_seasons(db):
    db.execute("INSERT INTO nhl_player_bio (player_id, birth_date) VALUES (7, '2000-06-15')")
    rt = SimpleNamespace(nhl_season="20262027", week_of=lambda d: 1, week=lambda n: _no_week())
    cand = SimpleNamespace(
        name="Martin Pospisil",
        team="CGY",
        yahoo_eligible=frozenset({"C", "RW", "Util"}),
        nhl_player_id=7,
        position="C",
        player_key="p.add",
    )
    drop = SimpleNamespace(
        name="John Gibson",
        team="DET",
        yahoo_eligible=frozenset({"G"}),
        nhl_player_id=8,
        position="G",
        player_key="p.drop",
    )
    got = add_story.profile_lines(db, rt, cand, drop, "2026-10-02")
    assert got[0] == "Martin Pospisil (CGY, C/RW, age 26)"
    assert got[1].startswith("  25-26: ") and got[2].startswith("  26-27: ")
    assert got[3] == "John Gibson (DET, G)"


def _no_week():
    raise KeyError("no next week")
