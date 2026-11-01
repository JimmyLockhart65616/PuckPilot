"""The numbers a drop tier is set by: this season's rank and VORP, and the rest of it."""

from __future__ import annotations

from puckpilot.league import DEFAULT_LEAGUE
from puckpilot.season import roster_stats
from puckpilot.season.roster import RosterPlayer, TeamRoster
from tests.conftest import add_player, add_skater_game

SEASON = "20262027"


def test_this_season_s_rank_counts_only_the_games_before_the_day(db):
    for pid, name in ((1, "Star"), (2, "Mid"), (3, "Depth")):
        add_player(db, pid, name, "C")
    for g in range(3):
        day = f"2026-10-1{g}"
        add_skater_game(db, 1, SEASON, 100 + g, date=day, goals=2, points=3, shots=5)
        add_skater_game(db, 2, SEASON, 200 + g, date=day, goals=1, points=1, shots=3)
        add_skater_game(db, 3, SEASON, 300 + g, date=day, shots=1)
    # Depth's big night comes after the cutoff: not this morning's season.
    add_skater_game(db, 3, SEASON, 399, date="2026-11-02", goals=5, points=5, shots=9)
    ranks = roster_stats.season_ranks(db, DEFAULT_LEAGUE, SEASON, "2026-11-01")
    assert ranks[1][0] < ranks[2][0] < ranks[3][0] and ranks[1][1] > ranks[2][1]
    after = roster_stats.season_ranks(db, DEFAULT_LEAGUE, SEASON, "2026-11-03")
    assert after[3][0] < after[2][0]


def _p(key, pid):
    return RosterPlayer(player_key=key, yahoo_id=key, name=key.title(), team="TOR",
                        primary_position="C", yahoo_eligible=frozenset({"C"}),
                        selected_slot="C", nhl_player_id=pid)  # fmt: skip


def test_each_row_says_his_season_and_his_place_by_the_rest_of_it():
    roster = TeamRoster(league_key="L", team_key="t", date="2026-11-01",
                        players=(_p("star", 1), _p("depth", 2), _p("rookie", 3)))  # fmt: skip
    keep = {"star": 400.0, "depth": 250.0, "rookie": 300.0}
    got = roster_stats.describe(roster, keep, {1: (12, 3.44), 2: (300, -1.2)})
    assert got["star"]["text"] == "Season #12, VORP +3.4 · Rest of season 400, 1st of 3"
    assert got["rookie"]["text"] == "No games this season · Rest of season 300, 2nd of 3"
    assert (got["depth"]["season_rank"], got["depth"]["rest_place"]) == (300, 3)


def test_places_read_as_english():
    assert [roster_stats._nth(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101, 111)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "101st",
        "111th",
    ]  # fmt: skip
