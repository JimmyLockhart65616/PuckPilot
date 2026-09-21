"""Tonight's lineup diff.

The value model is stubbed: `build_plan` only ever asks it what one game is
worth, and the blending behind that number is `waivers.blended_pg_value`, which
has its own tests. What matters here is who becomes a candidate, what the
optimizer is asked, and what the diff says to do in Yahoo.
"""

from __future__ import annotations

import pytest

from puckpilot.data import store
from puckpilot.season.authority import LineupAuthority
from puckpilot.season.goalies import StaticGoalieSource
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import LeagueRuntime, Week
from puckpilot.season.today import build_plan

DATE = "2026-10-07"
SEASON = "20262027"


class Values:
    """Per-game value straight from a dict."""

    def __init__(self, by_pid):
        self.by_pid = by_pid

    def per_game(self, pid, date):
        return float(self.by_pid.get(pid, 0.0))


def runtime(**over):
    slots = over.pop(
        "slots",
        [
            {"roster_position": {"position": p, "count": c, "is_starting_position": s}}
            for p, c, s in (
                ("C", 1, 1),
                ("RW", 1, 1),
                ("D", 1, 1),
                ("G", 1, 1),
                ("BN", 2, 0),
                ("IR", 1, 0),
            )
        ],
    )
    base = {
        "league_key": "999.l.1",
        "name": "T",
        "num_teams": 10,
        "scoring_type": "head",
        "season": "2026",
        "start_date": "2026-09-29",
        "end_date": "2027-03-28",
        "start_week": "1",
        "end_week": "25",
        "current_week": 2,
        "current_date": DATE,
        "playoff_start_week": "23",
        "num_playoff_teams": "8",
        "weekly_deadline": "intraday",
        "roster_type": "date",
        "waiver_type": "R",
        "waiver_time": "1",
        "uses_faab": "0",
        "max_adds": "65",
        "max_weekly_adds": "3",
        "min_games_played": "0",
        "roster_positions": slots,
    }
    base.update(over)
    return LeagueRuntime.from_payload(
        base, weeks=(Week(2, "2026-10-05", "2026-10-11"),), fetched_at="now"
    )


def player(key, name, pid, team, pos, slot, eligible=None, status="", editable=True):
    return RosterPlayer(
        player_key=key,
        yahoo_id=key.split(".")[-1],
        name=name,
        team=team,
        primary_position=pos,
        yahoo_eligible=frozenset(eligible or (pos, "Util")),
        selected_slot=slot,
        nhl_player_id=pid,
        status=status,
        is_editable=editable,
    )


def roster(*players):
    return TeamRoster(
        league_key="999.l.1", team_key="999.l.1.t.5", date=DATE, players=tuple(players)
    )


@pytest.fixture
def db_with_games(db):
    store.upsert_schedule_game(
        db,
        game_id=1,
        season=SEASON,
        game_type=2,
        game_date=DATE,
        start_time_utc=None,
        home_team="TOR",
        away_team="MTL",
    )
    db.commit()
    return db


def plan(db, rost, values, *, goalies=None, auth=None, date=DATE):
    return build_plan(
        db,
        runtime(),
        rost,
        Values(values),
        goalies or StaticGoalieSource(),
        date,
        manager="test",
        authority=auth or LineupAuthority(enabled=True, min_gain=0.0),
    )


# -- who is a candidate -----------------------------------------------------


def test_a_benched_player_whose_team_plays_is_started(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Plays", 1, "TOR", "C", "BN"),
            player("p.2", "Idle", 2, "VAN", "C", "C"),
        ),
        {1: 5.0, 2: 9.0},
    )
    assert [m.describe() for m in p.moves] == ["START Plays in C", "BENCH Idle  (was C)"]


def test_an_idle_starter_is_left_alone_when_nobody_needs_his_slot(db_with_games):
    """Benching every idle starter would mean eight pointless changes on a
    three-game Friday."""
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Idle D", 1, "VAN", "D", "D"),
            player("p.2", "Idle G", 2, "VAN", "G", "G"),
        ),
        {1: 5.0, 2: 5.0},
    )
    assert p.is_noop
    assert [x.name for x in p.idle] == ["Idle D", "Idle G"]


def test_a_player_ruled_out_is_never_started(db_with_games):
    p = plan(
        db_with_games,
        roster(player("p.1", "Hurt", 1, "TOR", "C", "BN", status="O")),
        {1: 99.0},
    )
    assert p.is_noop
    assert [x.name for x in p.out] == ["Hurt"]


def test_a_locked_player_is_reported_not_moved(db_with_games):
    """His game has started; the slot cannot be changed."""
    p = plan(
        db_with_games,
        roster(player("p.1", "Started", 1, "TOR", "C", "C", editable=False)),
        {1: 1.0},
    )
    assert p.is_noop
    assert [x.name for x in p.locked] == ["Started"]


def test_an_unmapped_player_is_flagged_and_left_alone(db_with_games):
    r = roster(player("p.1", "Call Up", None, "TOR", "C", "BN"))
    p = plan(db_with_games, r, {})
    assert p.is_noop
    assert any("not in the player map" in n for n in p.notes)


# -- goalies ----------------------------------------------------------------


def test_a_goalie_below_the_agreed_probability_is_not_started(db_with_games):
    g = StaticGoalieSource({DATE: {1: 0.2}})
    p = plan(
        db_with_games,
        roster(player("p.1", "Backup", 1, "TOR", "G", "BN")),
        {1: 9.0},
        goalies=g,
        auth=LineupAuthority(enabled=True, min_gain=0.0, min_goalie_p_start=0.5),
    )
    assert p.is_noop


def test_a_likely_starter_is_started_and_his_value_is_weighted(db_with_games):
    g = StaticGoalieSource({DATE: {1: 0.8}})
    p = plan(
        db_with_games,
        roster(player("p.1", "Starter", 1, "TOR", "G", "BN")),
        {1: 10.0},
        goalies=g,
    )
    assert [m.describe() for m in p.moves] == ["START Starter in G"]
    assert p.gain == pytest.approx(8.0)  # 10.0 x 0.8


def test_the_weekly_goalie_minimum_overrides_the_probability_floor(db_with_games):
    """Yahoo enforces it; falling short forfeits the categories."""
    rt = runtime(min_games_played="1")
    r = roster(player("p.1", "Any Goalie", 1, "TOR", "G", "BN"))
    p = build_plan(
        db_with_games,
        rt,
        r,
        Values({1: 1.0}),
        StaticGoalieSource({DATE: {1: 0.9}}),
        "2026-10-11",  # last day of the week
        manager="test",
        authority=LineupAuthority(enabled=True, min_gain=0.0),
        goalie_starts_so_far=0,
    )
    assert any("goalie minimum" in n for n in p.notes)


# -- the diff ---------------------------------------------------------------


def test_a_slot_change_is_a_move_not_a_start(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Flex", 1, "TOR", "C", "C", eligible=("C", "RW", "Util")),
            player("p.2", "Centre Only", 2, "TOR", "C", "BN", eligible=("C", "Util")),
        ),
        {1: 5.0, 2: 4.0},
    )
    described = [m.describe() for m in p.moves]
    assert "MOVE  Flex C -> RW" in described
    assert "START Centre Only in C" in described


def test_gain_is_the_difference_between_lineups_not_the_sum_of_what_moves(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Good", 1, "TOR", "C", "BN"),
            player("p.2", "Bad", 2, "TOR", "C", "C"),
        ),
        {1: 5.0, 2: 1.0},
    )
    assert p.gain == pytest.approx(4.0)


def test_an_already_optimal_lineup_produces_nothing(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Good", 1, "TOR", "C", "C"),
            player("p.2", "Bad", 2, "TOR", "C", "BN"),
        ),
        {1: 5.0, 2: 1.0},
    )
    assert p.is_noop
    assert "already optimal" in p.text()


def test_empty_slots_are_reported(db_with_games):
    p = plan(db_with_games, roster(player("p.1", "C", 1, "TOR", "C", "C")), {1: 5.0})
    assert set(p.empty_slots) == {"RW", "D", "G"}


# -- authority --------------------------------------------------------------


def test_a_change_below_the_agreed_threshold_is_left_alone(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Marginal", 1, "TOR", "C", "BN"),
            player("p.2", "Incumbent", 2, "TOR", "C", "C"),
        ),
        {1: 1.01, 2: 1.0},
        auth=LineupAuthority(enabled=True, min_gain=0.15),
    )
    assert p.is_noop
    assert any("below the agreed" in n for n in p.notes)


def test_without_standing_authority_the_plan_is_advice(db_with_games):
    p = plan(
        db_with_games,
        roster(player("p.1", "Plays", 1, "TOR", "C", "BN")),
        {1: 5.0},
        auth=LineupAuthority(enabled=False, min_gain=0.0),
    )
    assert p.moves
    assert p.within_authority is False
    assert "recommend only" in p.authority_reason


def test_too_many_changes_asks_instead_of_acting(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "A", 1, "TOR", "C", "BN"),
            player("p.2", "B", 2, "TOR", "RW", "BN"),
            player("p.3", "C", 3, "TOR", "D", "BN"),
        ),
        {1: 5.0, 2: 5.0, 3: 5.0},
        auth=LineupAuthority(enabled=True, min_gain=0.0, max_swaps_per_day=2),
    )
    assert len(p.moves) == 3
    assert p.within_authority is False
    assert "exceeds the agreed limit" in p.authority_reason


def test_never_bench_blocks_acting_but_still_reports(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Better", 1, "TOR", "C", "BN"),
            player("p.2", "Favourite", 2, "TOR", "C", "C"),
        ),
        {1: 9.0, 2: 1.0},
        auth=LineupAuthority(enabled=True, min_gain=0.0, never_bench=("Favourite",)),
    )
    assert p.moves
    assert p.within_authority is False
    assert "never to bench" in p.authority_reason


def test_questionable_players_start_only_to_fill_a_slot(db_with_games):
    healthy_wins = plan(
        db_with_games,
        roster(
            player("p.1", "Healthy", 1, "TOR", "C", "BN"),
            player("p.2", "Sore", 2, "TOR", "C", "BN", status="DTD"),
        ),
        {1: 1.0, 2: 9.0},
    )
    started = [m.player.name for m in healthy_wins.moves if m.is_start]
    assert started == ["Healthy"]

    alone = plan(
        db_with_games,
        roster(player("p.2", "Sore", 2, "TOR", "C", "BN", status="DTD")),
        {2: 9.0},
    )
    assert [m.player.name for m in alone.moves if m.is_start] == ["Sore"]


def test_questionable_players_can_be_benched_outright(db_with_games):
    p = plan(
        db_with_games,
        roster(player("p.1", "Sore", 1, "TOR", "C", "BN", status="DTD")),
        {1: 9.0},
        auth=LineupAuthority(enabled=True, min_gain=0.0, start_questionable="never"),
    )
    assert p.is_noop
    assert [x.name for x in p.out] == ["Sore"]


# -- churn ------------------------------------------------------------------


def test_a_flex_player_keeps_his_slot_when_either_would_do(db_with_games):
    """The assignment is indifferent between a multi-position player's slots,
    and an arbitrary choice displaces whoever holds the other one. Measured on
    the real roster this turned a 2-change evening into a 4-change one."""
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Flex", 1, "TOR", "C", "C", eligible=("C", "RW", "Util")),
            player("p.2", "Winger", 2, "TOR", "RW", "RW", eligible=("RW", "Util")),
        ),
        {1: 5.0, 2: 5.0},
    )
    assert p.is_noop


def test_an_idle_starter_is_only_displaced_when_his_slot_is_needed(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Idle RW", 1, "VAN", "RW", "RW", eligible=("RW", "Util")),
            player("p.2", "Idle D", 2, "VAN", "D", "D"),
            player("p.3", "Plays RW", 3, "TOR", "RW", "BN", eligible=("RW", "Util")),
        ),
        {1: 5.0, 2: 5.0, 3: 4.0},
    )
    described = [m.describe() for m in p.moves]
    assert "START Plays RW in RW" in described
    assert "BENCH Idle RW  (was RW)" in described
    assert not any("Idle D" in d for d in described)  # his slot was not wanted
