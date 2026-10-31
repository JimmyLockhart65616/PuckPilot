"""The whole league's record: transactions, standings, weekly totals, week-start rosters.

Payloads are Yahoo-shaped by hand (the shapes the 2026-10-31 read returned),
so the parsers are pinned without a network.
"""

from __future__ import annotations

from types import SimpleNamespace

from puckpilot.season import league_feed as lf
from puckpilot.season.roster import RosterPlayer, TeamRoster


def _player(key, full, team, pos, data):
    return {
        "player": [
            [
                {"player_key": key},
                {"name": {"full": full}},
                {"editorial_team_abbr": team},
                {"display_position": pos},
            ],  # fmt: skip
            {"transaction_data": data},
        ]
    }


TXNS = {
    "fantasy_content": {
        "league": [
            {"league_key": "L"},
            {
                "transactions": {
                    "count": 2,
                    "0": {
                        "transaction": [
                            {
                                "transaction_key": "L.tr.2",
                                "type": "add/drop",
                                "status": "successful",
                                "timestamp": "1761930000",
                            },  # fmt: skip
                            {
                                "players": {
                                    "count": 2,
                                    "0": _player(
                                        "p.1",
                                        "Pickup Goalie",
                                        "PHI",
                                        "G",
                                        [
                                            {
                                                "type": "add",
                                                "source_type": "freeagents",
                                                "destination_team_key": "L.t.3",
                                            }
                                        ],
                                    ),
                                    "1": _player(
                                        "p.2",
                                        "Cut Skater",
                                        "MIN",
                                        "D",
                                        {
                                            "type": "drop",
                                            "source_type": "team",
                                            "source_team_key": "L.t.3",
                                        },
                                    ),
                                }  # fmt: skip
                            },
                        ]
                    },
                    "1": {
                        "transaction": [
                            {
                                "transaction_key": "L.tr.1",
                                "type": "commish",
                                "status": "successful",
                                "timestamp": "1756000000",
                            },  # fmt: skip
                        ]
                    },
                }
            },
        ]
    }
}

STANDINGS = {
    "fantasy_content": {
        "league": [
            {"league_key": "L"},
            {
                "standings": [
                    {
                        "teams": {
                            "count": 1,
                            "0": {
                                "team": [
                                    [
                                        {"team_key": "L.t.3"},
                                        {"name": "Rivals"},
                                        {"number_of_moves": 9},
                                        {"number_of_trades": 0},
                                    ],  # fmt: skip
                                    {"team_points": {"total": "25"}},
                                    {
                                        "team_standings": {
                                            "rank": "2",
                                            "outcome_totals": {
                                                "wins": "25",
                                                "losses": "16",
                                                "ties": "3",
                                            },
                                        }
                                    },  # fmt: skip
                                ]
                            },
                        }
                    }
                ]
            },
        ]
    }
}


def _team(key, name, won, games, stats):
    return {
        "team": [
            [{"team_key": key}, {"name": name}],
            {
                "team_stats": {"stats": [{"stat": {"stat_id": k, "value": v}} for k, v in stats]},
                "team_points": {"total": won},
                "team_remaining_games": {"total": {"completed_games": games}},
            },
        ]
    }


BOARD = {
    "fantasy_content": {
        "league": [
            {"league_key": "L"},
            {
                "scoreboard": {
                    "0": {
                        "matchups": {
                            "count": 1,
                            "0": {
                                "matchup": {
                                    "status": "postevent",
                                    "0": {
                                        "teams": {
                                            "0": _team(
                                                "L.t.3",
                                                "Rivals",
                                                "9",
                                                49,
                                                [("1", "9"), ("26", ".909")],
                                            ),
                                            "1": _team(
                                                "L.t.5",
                                                "Us",
                                                "2",
                                                43,
                                                [("1", "18"), ("26", ".877")],
                                            ),
                                        }  # fmt: skip
                                    },
                                }
                            },
                        }
                    }
                }
            },
        ]
    }
}


def test_transactions_parse_into_one_row_per_player_moved():
    got = lf.parse_transactions(TXNS)
    assert [t.type for t in got] == ["add/drop", "commish"]
    add, drop = got[0].moves
    assert (add.action, add.team_key, add.name, add.positions, add.source_type) == (
        "add", "L.t.3", "Pickup Goalie", "G", "freeagents"
    )  # fmt: skip
    assert (drop.action, drop.team_key) == ("drop", "L.t.3")
    assert got[0].at.startswith("2025-") or got[0].at.startswith("2026-")


def test_standings_and_scoreboard_parse():
    (s,) = lf.parse_standings(STANDINGS)
    assert (s.team_key, s.rank, s.wins, s.losses, s.ties, s.moves) == ("L.t.3", 2, 25, 16, 3, 9)
    rivals, us = lf.parse_scoreboard(BOARD, week=4)
    assert (rivals.opponent_key, rivals.cats_won, rivals.completed_games) == ("L.t.5", 9.0, 49)
    assert us.stats == {"1": 18.0, "26": 0.877} and us.status == "postevent"


class _Source:
    def __init__(self):
        self.reads = []

    def transactions(self, league_key):
        self.reads.append("tx")
        return lf.parse_transactions(TXNS)

    def standings(self, league_key):
        self.reads.append("st")
        return lf.parse_standings(STANDINGS)

    def scoreboard(self, league_key, week):
        self.reads.append(f"sb{week}")
        return lf.parse_scoreboard(BOARD, week)

    def roster(self, team_key, date, player_map=None):
        self.reads.append(f"ro:{team_key}:{date}")
        p = RosterPlayer(player_key="p.9", yahoo_id="9", name="Any", team="TOR",
                         primary_position="C", yahoo_eligible=frozenset({"C"}),
                         selected_slot="C")  # fmt: skip
        return TeamRoster(league_key="L", team_key=team_key, date=date, players=(p,))


def test_a_sync_is_idempotent_and_reads_a_final_week_once(db):
    weeks = (SimpleNamespace(number=4, start="2026-10-19"),
             SimpleNamespace(number=5, start="2026-10-26"))  # fmt: skip
    runtime = SimpleNamespace(weeks=weeks)
    src = _Source()
    first = lf.sync(db, src, "L", runtime, "m", today="2026-10-31", player_map={"p.1": 77})
    assert first == {"transactions": 2, "standings": 1, "weeks": 2, "rosters": 2}
    src.reads.clear()
    again = lf.sync(db, src, "L", runtime, "m", today="2026-10-31")
    assert again["rosters"] == 0 and "sb4" not in src.reads  # final and saved: not re-read
    row = db.execute(
        "SELECT nhl_player_id FROM league_transactions WHERE player_key = 'p.1'"
    ).fetchone()
    assert row[0] == 77


def test_the_summary_says_who_is_adding_what(db):
    weeks = (SimpleNamespace(number=4, start="2026-10-19"),)
    lf.sync(db, _Source(), "L", SimpleNamespace(weeks=weeks), "m", today="2026-10-31")
    (team,) = lf.summary(db, "L")["teams"]
    assert (team["name"], team["record"], team["adds"], team["goalie_adds"]) == (
        "Rivals", "25-16-3", 1, 1
    )  # fmt: skip


def test_g3_scores_a_category_case_and_its_calibration():
    from puckpilot.engine.categories import CATALOG
    from puckpilot.season import g3

    sv = CATALOG["SV"]
    assert g3._outcome(sv, 120.0, 100.0) == 1.0 and g3._outcome(sv, 100.0, 100.0) == 0.5
    gaa = CATALOG["GAA"]
    assert g3._outcome(gaa, 2.1, 3.0) == 1.0  # lower is better
    case = g3.CatCase(1, "a", "b", "SV", "goalie", 90.0, 95.0, 120.0, 100.0, 0.4, 0.0, 1.0)
    rep = g3.G3Report([case], goalie_adds={(1, "a"): 1}, our_team="a")
    assert abs(rep.brier() - 0.36) < 1e-9
    assert rep.bias("SV")[0] == 30.0
    (streamed, _, _), (_, _, n_rest) = rep.streaming("SV")
    assert streamed == 30.0 and n_rest == 0  # no team-week without an add
    assert "G3 (live, whole league)" in rep.text()


def test_a_saved_roster_comes_back_with_positions(db):
    from puckpilot.season import g3
    from puckpilot.season.fetch import save_roster

    p = RosterPlayer(player_key="p.7", yahoo_id="7", name="Keeper", team="DET",
                     primary_position="G", yahoo_eligible=frozenset({"G"}), selected_slot="G",
                     nhl_player_id=42)  # fmt: skip
    save_roster(db, "m", TeamRoster(league_key="L", team_key="L.t.3", date="2026-10-19",
                                    players=(p,)))  # fmt: skip
    got = g3.saved_roster(db, "L", "L.t.3", "2026-10-19")
    assert [(q.position, q.nhl_player_id, q.selected_slot) for q in got.players] == [("G", 42, "G")]
    assert g3.saved_roster(db, "L", "L.t.3", "2026-10-26") is None
