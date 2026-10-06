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
        start_time_utc=f"{DATE}T23:00:00Z",
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


def test_a_locked_starter_keeps_his_slot_against_anyone_worth_more(db_with_games):
    """2026-10-04, 5:40 PM: Gibson's game had started and Wolf was worth more,
    so the plan benched Gibson - a move Yahoo refuses - and the run failed. A
    player whose game has started holds his slot: nobody moves him, nobody
    takes it."""
    g = StaticGoalieSource({DATE: {1: 1.0, 2: 1.0}})
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Playing Now", 1, "TOR", "G", "G", editable=False),
            player("p.2", "Worth More", 2, "MTL", "G", "BN"),
        ),
        {1: 1.0, 2: 9.0},
        goalies=g,
    )
    assert p.is_noop
    assert [x.name for x in p.locked] == ["Playing Now"]


def test_a_locked_starter_does_not_stop_the_rest_of_the_lineup(db_with_games):
    p = plan(
        db_with_games,
        roster(
            player("p.1", "Playing Now", 1, "TOR", "C", "C", editable=False),
            player("p.2", "Bench C", 2, "MTL", "C", "BN"),
            player("p.3", "Bench RW", 3, "MTL", "RW", "BN"),
        ),
        {1: 1.0, 2: 9.0, 3: 5.0},
    )
    # The C slot is held, so the better centre waits; the empty RW is filled.
    assert [m.describe() for m in p.moves] == ["START Bench RW in RW"]


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


def _minimum_plan(
    db, p_start, so_far, *, later=None, value=1.0, waived=None, date=DATE, steer=None
):
    """A one-goalie roster against a weekly minimum of one game."""
    over = {"min_games_played": "1"}
    if waived:
        over["week_has_enough_qualifying_days"] = {str(waived): 0}
    starts = {DATE: {1: p_start}}
    if later:
        for day in later:
            store.upsert_schedule_game(
                db,
                game_id=100 + int(day[-2:]),
                season=SEASON,
                game_type=2,
                game_date=day,
                start_time_utc=f"{day}T23:00:00Z",
                home_team="TOR",
                away_team="OTT",
            )
            starts[day] = {1: 0.9}
        db.commit()
    return build_plan(
        db,
        runtime(**over),
        roster(player("p.1", "Backup", 1, "TOR", "G", "BN")),
        Values({1: value}),
        StaticGoalieSource(starts),
        date,
        manager="test",
        authority=LineupAuthority(enabled=True, min_gain=0.0, min_goalie_p_start=0.5),
        goalie_starts_so_far=so_far,
        steer=steer,
    )


def test_the_weekly_goalie_minimum_overrides_the_probability_floor(db_with_games):
    """Yahoo enforces it; falling short forfeits the category. A 20% backup in
    an otherwise empty G slot is the only way left to reach it."""
    p = _minimum_plan(db_with_games, p_start=0.2, so_far=0)
    assert [m.describe() for m in p.moves] == ["START Backup in G"]
    assert any("goalie minimum" in n and "mandatory" in n for n in p.notes)


def test_a_forced_start_is_not_reported_as_a_million_point_night(db_with_games):
    p = _minimum_plan(db_with_games, p_start=0.2, so_far=0, value=10.0)
    assert p.gain == pytest.approx(2.0)  # 10.0 x 0.2, the bump excluded


def test_the_minimum_is_not_forced_while_later_games_cover_it(db_with_games):
    """Two likely games left in the week: tonight's 20% backup is a choice."""
    p = _minimum_plan(db_with_games, 0.2, 0, later=("2026-10-09", "2026-10-10"))
    assert p.is_noop
    assert not any("goalie minimum" in n for n in p.notes)


def test_a_minimum_already_met_forces_nothing(db_with_games):
    p = _minimum_plan(db_with_games, p_start=0.2, so_far=1)
    assert p.is_noop


def test_an_unknown_count_is_said_rather_than_guessed(db_with_games):
    """The old fallback counted goalie slot-days - two a day whether anyone
    played - and read the minimum as met by the second day."""
    p = _minimum_plan(db_with_games, p_start=0.2, so_far=None)
    assert p.is_noop
    assert any("not being checked" in n for n in p.notes)


def test_a_waived_week_has_no_minimum(db_with_games):
    p = _minimum_plan(db_with_games, p_start=0.2, so_far=0, waived=2)
    assert p.is_noop
    assert not any("minimum" in n for n in p.notes)


def test_a_forced_night_with_no_goalie_playing_says_so(db_with_games):
    p = _minimum_plan(db_with_games, p_start=0.9, so_far=0, date="2026-10-10")
    assert any("at risk" in n for n in p.notes)


def test_yahoos_count_for_another_week_is_refused():
    from puckpilot.season.today import yahoo_goalie_games

    r = TeamRoster(
        league_key="999.l.1",
        team_key="999.l.1.t.5",
        date=DATE,
        players=(),
        goalie_games=2,
        goalie_games_week=1,
    )
    assert yahoo_goalie_games(r, runtime(), DATE) is None  # DATE is in week 2
    same = TeamRoster(**{**r.__dict__, "goalie_games_week": 2})
    assert yahoo_goalie_games(same, runtime(), DATE) == 2


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


def test_a_busy_night_is_not_treated_as_a_malfunction(db_with_games):
    """Lineup moves are free and unlimited until each player's game starts, so
    a night wanting many changes is a busy schedule. An earlier cap of 4 would
    have refused to act on 17.6% of days."""
    p = plan(
        db_with_games,
        roster(
            player("p.1", "A", 1, "TOR", "C", "BN"),
            player("p.2", "B", 2, "TOR", "RW", "BN"),
            player("p.3", "C", 3, "TOR", "D", "BN"),
            player("p.4", "D", 4, "TOR", "G", "BN"),
        ),
        {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0},
        goalies=StaticGoalieSource({DATE: {4: 0.9}}),
    )
    assert len(p.moves) == 4
    assert p.within_authority is True


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


def test_nothing_to_do_does_not_claim_an_authority_that_was_never_granted(db_with_games):
    """ "Nothing to do" is trivially within authority, but telling someone who
    granted none that the tool "will act automatically" misdescribes what
    happens on the night there IS something to do."""
    r = roster(player("p.1", "Idle", 1, "VAN", "C", "C"))
    off = plan(db_with_games, r, {1: 5.0}, auth=LineupAuthority(enabled=False))
    assert off.is_noop
    assert "recommend only" in off.text()
    assert "will act automatically" not in off.text()

    on = plan(db_with_games, r, {1: 5.0}, auth=LineupAuthority(enabled=True, min_gain=0.0))
    assert "standing authority granted" in on.text()


# -- the lock ---------------------------------------------------------------


def test_the_first_lock_is_reported_as_the_real_deadline(db_with_games):
    """A daily league locks each player when his own game starts, so the
    practical deadline is the earliest of them."""
    p = plan(db_with_games, roster(player("p.1", "Plays", 1, "TOR", "C", "BN")), {1: 5.0})
    assert p.lock_utc
    assert p.lock_team in ("TOR", "MTL")
    assert "First lock" in p.text()


def test_the_deadline_renders_in_local_time(db_with_games):
    p = plan(db_with_games, roster(player("p.1", "Plays", 1, "TOR", "C", "BN")), {1: 5.0})
    assert ":" in p.deadline("America/Toronto")


def test_no_games_means_no_deadline(db_with_games):
    p = plan(db_with_games, roster(player("p.1", "Idle", 1, "VAN", "C", "C")), {1: 5.0})
    assert p.lock_utc == ""
    assert p.deadline() == ""


def test_the_chance_of_reaching_the_minimum_is_counted_exactly():
    from puckpilot.season.today import _p_at_least

    assert _p_at_least([0.9, 0.9], 1) == pytest.approx(0.99)
    assert _p_at_least([0.9, 0.9], 2) == pytest.approx(0.81)
    assert _p_at_least([0.5, 0.5, 0.5], 2) == pytest.approx(0.5)
    assert _p_at_least([], 1) == 0.0
    assert _p_at_least([0.3], 0) == 1.0


def test_one_likely_game_left_is_not_enough_to_skip_tonight(db_with_games):
    """The Saturday case the calendar-day rule got wrong: a 70% chance later
    in the week is a 30% chance of forfeiting the category."""
    p = build_plan(
        db_with_games,
        runtime(min_games_played="1"),
        roster(player("p.1", "Backup", 1, "TOR", "G", "BN")),
        Values({1: 1.0}),
        StaticGoalieSource({DATE: {1: 0.2}, "2026-10-09": {1: 0.7}}),
        DATE,
        manager="test",
        authority=LineupAuthority(enabled=True, min_gain=0.0),
        goalie_starts_so_far=0,
    )
    # No game on 10-09 in this fixture: the only later chance is none at all.
    assert [m.describe() for m in p.moves] == ["START Backup in G"]


# -- IR ---------------------------------------------------------------------


def _ir_runtime():
    """C1 RW1 D1 G1, two bench, one IR and two IR+: eight active spots."""
    return runtime(
        slots=[
            {"roster_position": {"position": p, "count": c, "is_starting_position": s}}
            for p, c, s in (
                ("C", 1, 1),
                ("RW", 1, 1),
                ("D", 1, 1),
                ("G", 1, 1),
                ("BN", 2, 0),
                ("IR", 1, 0),
                ("IR+", 2, 0),
            )
        ]
    )


def _ir_plan(db, *players, manage=True):
    return build_plan(
        db,
        _ir_runtime(),
        roster(*players),
        Values({1: 5.0, 2: 3.0, 3: 4.0}),
        StaticGoalieSource(),
        DATE,
        manager="test",
        authority=LineupAuthority(enabled=True, min_gain=0.0, manage_ir=manage),
    )


def test_an_out_player_moves_to_a_free_ir_slot_rather_than_the_bench(db_with_games):
    """Sanderson's case: out, IR+-eligible, sitting in an active D slot. Benching
    him keeps a roster spot dead; IR+ frees it."""
    p = _ir_plan(
        db_with_games,
        player("p.1", "Hurt D", 1, "TOR", "D", "D", eligible=("D", "IR+", "Util"), status="O"),
        player("p.2", "Spare D", 2, "TOR", "D", "BN"),
    )
    assert [m.describe() for m in p.ir_moves] == ["IR    Hurt D D -> IR+"]
    # His slot is freed by the IR move, so nobody has to bench him first.
    assert [m.describe() for m in p.moves] == ["START Spare D in D"]
    assert p.ir_within_authority is True


def test_the_most_restrictive_ir_slot_is_used_first(db_with_games):
    p = _ir_plan(
        db_with_games,
        player("p.1", "Long Term", 1, "VAN", "C", "C", eligible=("C", "IR", "IR+"), status="IR"),
    )
    assert [m.to_slot for m in p.ir_moves] == ["IR"]


def test_no_ir_move_without_eligibility_or_room(db_with_games):
    full = _ir_plan(
        db_with_games,
        player("p.1", "Out A", 1, "VAN", "C", "C", eligible=("C", "IR+"), status="O"),
        player("p.2", "On IR", 2, "VAN", "RW", "IR+", eligible=("RW", "IR+"), status="O"),
        player("p.3", "On IR 2", 3, "VAN", "D", "IR+", eligible=("D", "IR+"), status="O"),
    )
    assert full.ir_moves == ()
    ineligible = _ir_plan(
        db_with_games, player("p.1", "Out B", 1, "VAN", "C", "C", eligible=("C",), status="O")
    )
    assert ineligible.ir_moves == ()


def test_a_day_to_day_player_is_never_sent_to_ir(db_with_games):
    """DTD players play most nights; IR is for players who cannot."""
    p = _ir_plan(
        db_with_games,
        player("p.1", "Sore", 1, "VAN", "C", "C", eligible=("C", "IR+"), status="DTD"),
    )
    assert p.ir_moves == ()


def test_a_healthy_player_left_in_ir_is_activated_when_there_is_room(db_with_games):
    p = _ir_plan(
        db_with_games,
        player("p.1", "Back", 1, "VAN", "C", "IR+", eligible=("C", "Util")),
    )
    assert [m.describe() for m in p.ir_moves] == ["ACTIVATE Back IR+ -> BN"]
    assert p.ir_alerts == ()


def test_an_activated_player_with_a_game_goes_straight_into_the_lineup(db_with_games):
    p = _ir_plan(
        db_with_games,
        player("p.1", "Back", 1, "TOR", "C", "IR+", eligible=("C", "Util")),
    )
    assert p.ir_moves == ()
    assert [m.describe() for m in p.moves] == ["ACTIVATE Back IR+ -> C"]


def test_an_illegal_ir_with_no_room_is_an_alert_not_a_move(db_with_games):
    """Yahoo refuses every add and drop while it stands, so it is said first."""
    healthy = [player(f"p.{i}", f"Body {i}", 10 + i, "VAN", "C", "BN") for i in range(8)]
    p = _ir_plan(
        db_with_games,
        *healthy,
        player("p.9", "Back", 1, "VAN", "C", "IR+", eligible=("C", "Util")),
    )
    assert p.ir_moves == ()
    assert len(p.ir_alerts) == 1 and "drop has to come first" in p.ir_alerts[0]


def test_ir_moves_are_recommendations_until_agreed(db_with_games):
    p = _ir_plan(
        db_with_games,
        player("p.1", "Hurt", 1, "VAN", "D", "D", eligible=("D", "IR+"), status="O"),
        manage=False,
    )
    assert p.ir_moves and p.ir_within_authority is False


# -- a goalie below the floor, into a slot that would otherwise be empty ------


def _floor_plan(db, goalies, values, fill=True):
    """One G slot; goalies as (key, name, pid, p_start)."""
    return build_plan(
        db,
        runtime(),
        roster(*(player(k, n, pid, "TOR", "G", "BN") for k, n, pid, _ in goalies)),
        Values(values),
        StaticGoalieSource({DATE: {pid: p for _, _, pid, p in goalies}}),
        DATE,
        manager="test",
        authority=LineupAuthority(
            enabled=True, min_gain=0.0, min_goalie_p_start=0.5, fill_empty_goalie_slot=fill
        ),
    )


def test_a_goalie_below_the_floor_fills_a_slot_that_would_otherwise_be_empty(db_with_games):
    """Not starting scores nothing - the same as the empty slot - so the floor
    costs a chance and protects nothing."""
    p = _floor_plan(db_with_games, [("g.1", "Backup", 1, 0.3)], {1: 5.0})
    assert [m.describe() for m in p.moves] == ["START Backup in G"]
    assert "would otherwise be empty" in p.playing[0].note


def test_without_the_agreement_the_floor_still_holds(db_with_games):
    p = _floor_plan(db_with_games, [("g.1", "Backup", 1, 0.3)], {1: 5.0}, fill=False)
    assert p.is_noop


def test_the_floor_still_decides_between_goalies_for_one_slot(db_with_games):
    """A 45% goalie with the bigger projection does not take the slot from a
    60% one - that would be removing the floor, which is not what was agreed."""
    p = _floor_plan(
        db_with_games,
        [("g.1", "Likely", 1, 0.6), ("g.2", "Long Shot Star", 2, 0.45)],
        {1: 2.0, 2: 9.0},
    )
    assert [m.describe() for m in p.moves] == ["START Likely in G"]
    assert [x.name for x in p.idle] == ["Long Shot Star"]


# -- steered by this week's odds ------------------------------------------------


def _steer(leverage=None, rates=None, ours=None, theirs=None, cats=None):
    from puckpilot.engine.categories import resolve
    from puckpilot.season.odds import OddsModel, Side
    from puckpilot.season.week import Steer

    return Steer(
        leverage=leverage or {},
        rates=rates or {},
        model=OddsModel(),
        cats=cats or (resolve("G"), resolve("HIT")),
        ours=ours or Side(),
        theirs=theirs or Side(),
    )


def _one_c_night(db, steer, auth=None):
    """Two centres with a game, one C slot: a bench call."""
    return build_plan(
        db,
        runtime(),
        roster(
            player("p.1", "Scorer", 1, "TOR", "C", "BN"),
            player("p.2", "Hitter", 2, "MTL", "C", "BN"),
        ),
        Values({1: 3.0, 2: 1.0}),
        StaticGoalieSource(),
        DATE,
        manager="test",
        authority=auth or LineupAuthority(enabled=True, min_gain=0.0),
        steer=steer,
    )


RATES = {1: {"goals": 0.6, "hits": 0.5}, 2: {"goals": 0.1, "hits": 3.0}}


def test_steered_a_bench_call_goes_to_the_player_the_close_category_needs(db_with_games):
    """Goals decided, hits level: the hitter's game is worth more this week
    than the scorer's, whatever their season values say."""
    plain = _one_c_night(db_with_games, None)
    steered = _one_c_night(db_with_games, _steer({"goals": 0.0, "hits": 0.05}, RATES))
    assert [m.describe() for m in plain.moves] == ["START Scorer in C"]
    assert [m.describe() for m in steered.moves] == ["START Hitter in C"]
    assert any(n.startswith("By this week's odds: Hitter start over Scorer") for n in steered.notes)


def test_steered_a_decided_week_lines_up_as_unsteered(db_with_games):
    p = _one_c_night(db_with_games, _steer({"goals": 0.0, "hits": 0.0}, RATES))
    assert [m.describe() for m in p.moves] == ["START Scorer in C"]
    assert not any(n.startswith("By this week's odds") for n in p.notes)


def test_steered_moves_are_not_refused_by_a_floor_in_other_units(db_with_games):
    """The agreed floor is in per-game value units; steered values are expected
    categories, a hundredth the size. Unscaled, it would refuse every one."""
    p = _one_c_night(
        db_with_games,
        _steer({"goals": 0.0, "hits": 0.05}, RATES),
        auth=LineupAuthority(enabled=True, min_gain=0.15),
    )
    assert [m.describe() for m in p.moves] == ["START Hitter in C"]


def _settled_goalie_week():
    """Wins and saves locked up, save percentage a narrow lead."""
    from puckpilot.engine.categories import resolve
    from puckpilot.season.odds import Side

    return _steer(
        rates={1: {"wins": 0.5, "saves": 24.0, "shots_against": 26.5}},  # .906
        ours=Side(banked={"wins": 5.0, "saves": 230.0, "shots_against": 250.0}),  # .920
        theirs=Side(banked={"wins": 1.0, "saves": 91.0, "shots_against": 100.0}),  # .910
        cats=(resolve("W"), resolve("SV"), resolve("SV%")),
    )


def test_steered_a_start_that_can_only_cost_the_open_category_sits(db_with_games):
    g = StaticGoalieSource({DATE: {1: 0.95}})
    p = plan(
        db_with_games, roster(player("p.1", "Starter", 1, "TOR", "G", "BN")), {1: 10.0}, goalies=g
    )
    assert [m.describe() for m in p.moves] == ["START Starter in G"]
    steered = build_plan(
        db_with_games,
        runtime(),
        roster(player("p.1", "Starter", 1, "TOR", "G", "BN")),
        Values({1: 10.0}),
        g,
        DATE,
        manager="test",
        authority=LineupAuthority(enabled=True, min_gain=0.0),
        steer=_settled_goalie_week(),
    )
    assert steered.is_noop
    assert any("Starter sits" in n for n in steered.notes)


def test_steering_never_overrides_the_weekly_goalie_minimum(db_with_games):
    p = _minimum_plan(db_with_games, p_start=0.95, so_far=0, steer=_settled_goalie_week())
    assert [m.describe() for m in p.moves] == ["START Backup in G"]
