"""`draft preflight`: each check against the failure it exists to catch.

Every one of these failures produces a plausible-looking board, which is the
whole reason for a check: nothing here would crash on draft night.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from puckpilot import preflight as pf
from puckpilot.data import store
from puckpilot.draft.board import DraftBoard
from puckpilot.draft.engine import DraftRules, Universe
from puckpilot.engine.categories import Category
from puckpilot.engine.valuation import LeagueShape
from puckpilot.league import LeagueConfig

SHAPE = LeagueShape(n_teams=4, slots=(("C", 1), ("D", 1), ("G", 1)), util_slots=0, bench_slots=1)
NOW = dt.datetime(2026, 9, 17, 12, 0)
SEASON = "20262027"


def _league(**kw):
    base = dict(
        name="Test",
        shape=SHAPE,
        skater_cats=(Category("goals", "G", "skater"),),
        goalie_cats=(Category("wins", "W", "goalie"),),
        n_keepers=1,
        keepers_by_season={SEASON: ("Alpha C", "Beta D", "Gamma G", "Delta C")},
        keeper_owners_by_season={SEASON: {2: ("Beta D",)}},
    )
    base.update(kw)
    return LeagueConfig(**base)


PLAYERS = [
    (1, "Alpha C", "C"),
    (2, "Beta D", "D"),
    (3, "Gamma G", "G"),
    (4, "Delta C", "C"),
    (5, "Echo D", "D"),
    (6, "Foxtrot G", "G"),
    (7, "Golf C", "C"),
    (8, "Hotel D", "D"),
]


def _players(db):
    for pid, name, pos in PLAYERS:
        store.upsert_player(db, pid, name, pos, "AAA")


def _universe():
    df = pd.DataFrame.from_dict(
        {
            pid: {
                "name": name,
                "position": pos,
                "team": "AAA",
                "vorp": 10.0 - pid,
                "z_total": 10.0 - pid,
                "adp_rank": float(pid),
                "source": "projected",
            }
            for pid, name, pos in PLAYERS
        },
        orient="index",
    )
    df.index.name = "player_id"
    return Universe(df)


RULES = DraftRules(
    shape=SHAPE, rounds=3, caps={"C": 2, "D": 2, "G": 2}, mins={"C": 1, "D": 1, "G": 1}
)


def _board(keepers=None, placement="last"):
    return DraftBoard(
        _universe(),
        RULES,
        my_seat=2,
        keepers=keepers if keepers is not None else {0: [1], 1: [4], 2: [2], 3: [3]},
        keeper_placement=placement,
    )


# ---- league file -----------------------------------------------------------


def test_a_league_file_that_does_not_load_is_a_fail(tmp_path):
    from puckpilot.league import load_league

    check, league = pf.check_league_file(tmp_path / "missing.toml", load_league)
    assert check.status == pf.FAIL and league is None


def test_roster_minimums_that_drift_from_the_league_fail():
    board = _board()
    assert pf.check_roster_rules(_league(), RULES, board, 2).status == pf.PASS
    wrong = DraftRules(shape=SHAPE, rounds=3, mins={"C": 2, "D": 1, "G": 1})
    assert pf.check_roster_rules(_league(), wrong, board, 2).status == pf.FAIL


# ---- the pick order ------------------------------------------------------------


def test_a_plain_snake_passes_and_lists_our_picks():
    check = pf.check_pick_sequence(_board(), _league(), 2)
    assert check.status == pf.PASS
    assert "#3," in check.lines[1]  # seat 2 picks third in round 1


def test_a_board_that_opens_on_the_wrong_seat_fails():
    """The real bug: keepers modelled in the wrong rounds put seat 5 on the
    clock at pick 1."""
    board = _board()
    board.slots = board.slots[1:] + board.slots[:1]
    board.slot_numbers = board.slot_numbers[1:] + board.slot_numbers[:1]
    assert pf.check_pick_sequence(board, _league(), 2).status == pf.FAIL


def test_first_placement_is_flagged_for_confirmation():
    check = pf.check_pick_sequence(_board(placement="first"), _league(), 2)
    assert check.status == pf.WARN


# ---- keepers ------------------------------------------------------------------


def test_keepers_all_resolved_and_all_slots_declared_pass(db):
    _players(db)
    assert pf.check_keepers(db, _league(), SEASON, _universe()).status == pf.PASS


def test_an_unmatched_keeper_fails(db):
    _players(db)
    league = _league(keepers_by_season={SEASON: ("Alpha C", "Nobody", "Gamma G", "Delta C")})
    check = pf.check_keepers(db, league, SEASON, _universe())
    assert check.status == pf.FAIL and "Nobody" in check.lines[0]


def test_undeclared_keeper_slots_warn(db):
    _players(db)
    league = _league(keepers_by_season={SEASON: ("Alpha C",)})
    check = pf.check_keepers(db, league, SEASON, _universe())
    assert check.status == pf.WARN and "3 of 4 keeper slots undeclared" in check.lines[0]


def test_no_keeper_list_at_all_fails(db):
    _players(db)
    assert pf.check_keepers(db, _league(keepers_by_season={}), SEASON, _universe()).status == (
        pf.FAIL
    )


def test_our_seat_undeclared_fails(db):
    _players(db)
    check = pf.check_keeper_owners(db, _league(), SEASON, seat=1)
    assert check.status == pf.FAIL and "seat 1 (yours)" in check.detail


def test_our_seat_declared_passes(db):
    _players(db)
    assert pf.check_keeper_owners(db, _league(), SEASON, seat=2).status == pf.PASS


def test_an_owner_claiming_a_player_off_the_list_fails(db):
    _players(db)
    league = _league(keeper_owners_by_season={SEASON: {2: ("Echo D",)}})
    assert pf.check_keeper_owners(db, league, SEASON, seat=2).status == pf.FAIL


# ---- keeper history -----------------------------------------------------------


def _saved(**manager):
    m = {
        "nickname": "Rival",
        "seat": 1,
        "continuing": [{"name": "Alpha C", "times_kept": 1}],
        "open_slots": 0,
        "candidates": [],
    }
    m.update(manager)
    return {"season": SEASON, "derived_at": "now", "managers": [m]}


def test_no_saved_history_is_informational():
    assert pf.check_keeper_history(_league(), SEASON, None).status == pf.INFO


def test_a_contract_missing_from_the_list_warns():
    saved = _saved(continuing=[{"name": "Echo D", "times_kept": 2}])
    check = pf.check_keeper_history(_league(), SEASON, saved)
    assert check.status == pf.WARN
    assert any("Echo D" in line and "NOT in the keeper list" in line for line in check.lines)


def test_an_open_slot_names_the_likeliest_keep_still_available():
    saved = _saved(open_slots=1, candidates=[{"name": "Golf C", "adp": 4}])
    check = pf.check_keeper_history(_league(), SEASON, saved)
    assert any("Golf C (ADP 4)" in line for line in check.lines)


def test_an_open_slot_filled_by_a_listed_candidate_is_quiet():
    league = _league(keepers_by_season={SEASON: ("Alpha C", "Golf C")})
    saved = _saved(open_slots=1, candidates=[{"name": "Golf C", "adp": 4}])
    check = pf.check_keeper_history(league, SEASON, saved)
    assert check.status == pf.PASS, check.lines


def test_a_listed_name_with_no_contract_and_no_open_slot_warns():
    league = _league(keepers_by_season={SEASON: ("Alpha C", "Hotel D")})
    saved = _saved(candidates=[{"name": "Hotel D", "adp": 50}])  # team is full
    check = pf.check_keeper_history(league, SEASON, saved)
    assert check.status == pf.WARN and "Hotel D" in check.lines[-1]


# ---- Yahoo data and local data ---------------------------------------------------


def _map(db, key, league, nhl, rank, updated="2026-09-16 12:00:00"):
    db.execute(
        "INSERT INTO yahoo_player_map (player_key, league_key, full_name, nhl_player_id,"
        " adp_rank, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (key, league, f"P{key}", nhl, rank, updated),
    )


def test_an_empty_or_missing_player_map_fails(db):
    assert pf.check_playermap(db, None, NOW).status == pf.FAIL
    assert pf.check_playermap(db, "477.l.1", NOW).status == pf.FAIL


def test_a_stale_player_map_warns(db):
    _map(db, "477.p.1", "477.l.1", 1, 1, updated="2026-09-01 00:00:00")
    check = pf.check_playermap(db, "477.l.1", NOW)
    assert check.status == pf.WARN and any("days ago" in line for line in check.lines)


def test_a_fresh_fully_resolved_map_passes(db):
    _map(db, "477.p.1", "477.l.1", 1, 1)
    assert pf.check_playermap(db, "477.l.1", NOW).status == pf.PASS


def test_no_adp_fails(db):
    check, key = pf.check_adp(db, None)
    assert check.status == pf.FAIL and key is None


def test_missing_schedule_fails_and_missing_roster_sync_warns(db):
    assert pf.check_data_freshness(db, SEASON, NOW).status == pf.FAIL
    store.upsert_schedule_game(
        db,
        game_id=1,
        season=SEASON,
        game_type=2,
        game_date="2026-10-07",
        start_time_utc=None,
        home_team="AAA",
        away_team="BBB",
    )
    assert pf.check_data_freshness(db, SEASON, NOW).status == pf.WARN
    store.set_meta(db, "rosters:20262027", "32 teams")
    db.execute("UPDATE sync_meta SET updated_at = '2026-09-17 00:00:00'")
    assert pf.check_data_freshness(db, SEASON, NOW).status == pf.PASS


def test_the_probe_never_fails_the_preflight():
    def boom():
        raise RuntimeError("offline")

    assert pf.check_yahoo_probe(boom).status == pf.INFO
    assert pf.check_yahoo_probe(None).status == pf.INFO


def test_any_fail_fails_the_report():
    report = pf.PreflightReport([pf.Check("a", pf.PASS, ""), pf.Check("b", pf.WARN, "")])
    assert not report.failed and "READY" in report.text
    report.checks.append(pf.Check("c", pf.FAIL, "boom"))
    assert report.failed and "NOT READY" in report.text


def test_projection_coverage_names_priced_players_we_cannot_see(db):
    board = _board()
    _map(db, "477.p.1", "477.l.1", 7, 1)  # on the board
    _map(db, "477.p.2", "477.l.1", None, 2)  # nowhere
    check = pf.check_projection_coverage(db, "477.l.1", board)
    assert check.status == pf.WARN and "NOT ON THE BOARD" in check.lines[0]
