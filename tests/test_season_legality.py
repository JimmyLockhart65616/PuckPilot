"""Roster legality: every state Yahoo refuses an add from, and the drop that fixes it.

On 2026-10-19 a defenceman on IR+ was cleared to play with every active spot
full. Yahoo refused every add ("Error #845") for twelve days while the run
printed "a drop has to come first" and nothing made one. These pin down the
mechanism that does: which drop, when none is needed, and when it must never
be made - in a league shaped like that one and in one that is not.
"""

from __future__ import annotations

from puckpilot.draft.engine import DraftRules
from puckpilot.season import legality
from puckpilot.season.moves import MoveKind
from puckpilot.season.roster import RosterPlayer, TeamRoster
from puckpilot.season.settings import LeagueRuntime, Week

DATE = "2026-10-26"


def runtime(slots):
    base = {
        "league_key": "999.l.1",
        "name": "T",
        "num_teams": 12,
        "scoring_type": "head",
        "season": "2026",
        "start_date": "2026-09-29",
        "end_date": "2027-03-28",
        "start_week": "1",
        "end_week": "25",
        "current_week": 5,
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
        "min_games_played": "3",
        "roster_positions": [
            {"roster_position": {"position": p, "count": c, "is_starting_position": int(p != "BN")}}
            for p, c in slots
        ],
    }
    return LeagueRuntime.from_payload(
        base, weeks=(Week(5, "2026-10-26", "2026-11-01"),), fetched_at="now"
    )


# Four active spots - C, D, G, one bench - and an IR and an IR+.
SMALL = runtime([("C", 1), ("D", 1), ("G", 1), ("BN", 1), ("IR", 1), ("IR+", 1)])
# No IR+ at all, an NA slot instead: a different league's shape.
ROTO = runtime([("C", 2), ("LW", 2), ("D", 2), ("G", 1), ("BN", 2), ("IR", 2), ("NA", 1)])
RULES = DraftRules(mins={"C": 1, "D": 1, "G": 1}, caps={"C": 9, "D": 9, "G": 9})


def p(key, pos, slot, eligible=(), status="", pid=None, protected=False, undroppable=False):
    return RosterPlayer(
        player_key=key,
        yahoo_id=key,
        name=key.title(),
        team="OTT",
        primary_position=pos,
        yahoo_eligible=frozenset(eligible or (pos, "Util")),
        selected_slot=slot,
        nhl_player_id=pid if pid is not None else abs(hash(key)) % 10_000,
        status=status,
        keeper_protected=protected,
        is_undroppable=undroppable,
    )


def team(*players):
    return TeamRoster(league_key="999.l.1", team_key="999.l.1.t.5", date=DATE, players=players)


# The 2026-10-19 shape: four active, all full, and a healthy D left in IR+.
CENTRE = p("centre", "C", "C")
STREAMER = p("streamer", "C", "BN")
DEFENCE = p("defence", "D", "D")
GOALIE = p("goalie", "G", "G")
BACK = p("back", "D", "IR+", eligible=("D", "Util"))  # cleared: IR+ no longer listed
KEEP = {"centre": 300.0, "streamer": 40.0, "defence": 250.0, "goalie": 200.0, "back": 280.0}


def test_a_cleared_player_in_ir_is_a_violation_and_an_injured_one_is_not():
    hurt = p("hurt", "D", "IR+", eligible=("D", "IR+", "Util"), status="IR")
    assert legality.check(SMALL, team(CENTRE, DEFENCE, GOALIE, hurt)) == ()
    got = legality.check(SMALL, team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK))
    assert [v.kind for v in got] == [legality.IR_INELIGIBLE]
    assert "Back" in got[0].message and "drop has to come first" in got[0].message


def test_a_full_roster_drops_the_least_valuable_legal_player():
    res = legality.resolve(SMALL, team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK), KEEP, RULES)
    assert not res.legal
    assert [(f.player.player_key, f.drop.player_key) for f in res.transactions] == [
        ("back", "streamer")
    ]


def test_a_free_spot_needs_no_drop_at_all():
    """The lineup brings him back; nothing to approve."""
    res = legality.resolve(SMALL, team(CENTRE, DEFENCE, GOALIE, BACK), KEEP, RULES)
    assert [f.drop for f in res.fixes] == [None] and res.transactions == ()


def test_a_free_spot_is_said_and_no_drop_is_asked_for():
    """2026-11-01: the drop that freed a spot was made at 07:01, and at 07:02
    the run still said "there is no free spot: a drop has to come first"."""
    (v,) = legality.check(SMALL, team(CENTRE, DEFENCE, GOALIE, BACK))
    assert "A spot is free" in v.message and "drop has to come first" not in v.message
    # A player going onto IR today frees his spot, counted as `resolve` counts it.
    hurt = p("streamer", "C", "BN", eligible=("C", "IR", "Util"), status="O")
    (v,) = legality.check(SMALL, team(CENTRE, hurt, DEFENCE, GOALIE, BACK))
    assert "A spot is free" in v.message
    res = legality.resolve(SMALL, team(CENTRE, hurt, DEFENCE, GOALIE, BACK), KEEP, RULES)
    assert [f.drop for f in res.fixes] == [None]


def test_two_coming_back_to_one_free_spot_need_a_drop_for_the_other():
    second = p("second", "C", "IR", eligible=("C", "Util"))
    got = legality.check(SMALL, team(CENTRE, DEFENCE, GOALIE, BACK, second))
    assert len(got) == 2
    assert all("1 active spot is free for the 2 players coming back" in v.message for v in got)


def test_the_returning_player_is_released_when_he_is_worth_least():
    cheap_back = {**KEEP, "back": 10.0}
    res = legality.resolve(SMALL, team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK), cheap_back, RULES)
    (fix,) = res.transactions
    assert fix.releases and fix.drop.player_key == "back"


def test_a_protected_keeper_is_never_released_or_dropped():
    keeper_back = p("back", "D", "IR+", eligible=("D", "Util"), protected=True)
    keeper_streamer = p("streamer", "C", "BN", protected=True)
    cheap_back = {**KEEP, "back": 10.0}
    res = legality.resolve(
        SMALL, team(CENTRE, keeper_streamer, DEFENCE, GOALIE, keeper_back), cheap_back, RULES
    )
    (fix,) = res.transactions
    assert not fix.releases and fix.drop.player_key not in ("streamer", "back")


def test_out_na_and_held_players_are_never_the_drop():
    out = p("streamer", "C", "BN", status="O")
    res = legality.resolve(SMALL, team(CENTRE, out, DEFENCE, GOALIE, BACK), KEEP, RULES)
    assert res.transactions[0].drop.player_key != "streamer"
    held = legality.resolve(
        SMALL, team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK), KEEP, RULES, hold={"streamer"}
    )
    assert held.transactions[0].drop.player_key != "streamer"


def test_roster_minimums_are_kept():
    """The only goalie is never the make-room drop, however cheap."""
    cheap_goalie = {**KEEP, "goalie": 1.0}
    res = legality.resolve(
        SMALL, team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK), cheap_goalie, RULES
    )
    assert res.transactions[0].drop.player_key == "streamer"


def test_no_legal_drop_is_said_not_guessed():
    keeper = {k: v for k, v in KEEP.items()}
    everyone = team(
        p("centre", "C", "C", protected=True),
        p("streamer", "C", "BN", protected=True),
        p("defence", "D", "D", protected=True),
        p("goalie", "G", "G", protected=True),
        p("back", "D", "IR+", eligible=("D", "Util"), protected=True),
    )
    res = legality.resolve(SMALL, everyone, keeper, RULES)
    assert res.transactions == () and res.unresolved and "cannot come back" in res.unresolved[0]


def test_two_returning_players_get_two_drops_most_valuable_placed_first():
    second = p("second", "C", "IR", eligible=("C", "Util"))
    roster = team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK, second)
    keep = {**KEEP, "second": 30.0}
    res = legality.resolve(SMALL, roster, keep, RULES)
    assert [f.player.player_key for f in res.fixes] == ["back", "second"]
    # Back takes the streamer's spot; Second is worth least of all, so goes himself.
    assert res.fixes[0].drop.player_key == "streamer"
    assert res.fixes[1].releases


def test_an_approved_drop_is_kept_while_it_is_on_the_roster():
    res = legality.resolve(
        SMALL,
        team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK),
        KEEP,
        RULES,
        promised={"back": "goalie"},
    )
    assert res.transactions[0].drop.player_key == "goalie"


def test_the_resolved_roster_is_legal():
    roster = team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK)
    res = legality.resolve(SMALL, roster, KEEP, RULES)
    after = legality.as_resolved(roster, res)
    assert {q.player_key for q in after.players} == {"centre", "defence", "goalie", "back"}
    assert next(q for q in after.players if q.player_key == "back").selected_slot == "BN"
    assert legality.check(SMALL, after) == ()


def test_a_league_without_ir_plus_and_with_na_works_the_same():
    """A called-up prospect left in NA is the same violation in a roto league."""
    players = [
        p("c1", "C", "C"),
        p("c2", "C", "C"),
        p("l1", "LW", "LW"),
        p("l2", "LW", "LW"),
        p("d1", "D", "D"),
        p("d2", "D", "D"),
        p("g1", "G", "G"),
        p("b1", "LW", "BN"),
        p("b2", "C", "BN"),
        p("kid", "C", "NA", eligible=("C", "Util")),  # no longer NA-eligible
    ]
    keep = {q.player_key: 100.0 for q in players} | {"b2": 5.0}
    res = legality.resolve(ROTO, team(*players), keep, DraftRules())
    assert [(f.player.player_key, f.drop.player_key) for f in res.transactions] == [("kid", "b2")]


def test_an_over_cap_active_roster_needs_a_drop():
    """Brought back by hand into a full roster: one too many active."""
    roster = team(CENTRE, STREAMER, DEFENCE, GOALIE, p("extra", "C", "BN"))
    keep = {**KEEP, "extra": 20.0}
    (v,) = legality.check(SMALL, roster)
    assert v.kind == legality.OVER_ACTIVE_CAP
    (fix,) = legality.resolve(SMALL, roster, keep, RULES).transactions
    assert fix.player is None and fix.drop.player_key == "extra"


def test_ir_changes_and_open_spots_are_unchanged_by_the_move():
    """`today.ir_changes` now lives here; its contract did not move."""
    from puckpilot.season.today import ir_changes

    hurt = p("hurt", "D", "D", eligible=("D", "IR+", "Util"), status="O")
    moves, alerts = ir_changes(SMALL, team(CENTRE, hurt, GOALIE, STREAMER))
    assert [m.describe() for m in moves] == ["IR    Hurt D -> IR+"] and alerts == []
    assert legality.open_roster_spots(SMALL, team(CENTRE, hurt, GOALIE, STREAMER)) == 1


def test_the_card_head_says_what_the_move_is():
    assert MoveKind.ACTIVATE.head("Back", "Streamer", "IR+") == (
        "Make room: drop Streamer so Back comes off IR+"
    )
    assert MoveKind.ACTIVATE.head("Back", "Back", "IR+").startswith("Release Back from IR+")
    assert MoveKind.ADD_DROP.head("Pickup", "") == "Add Pickup"
    assert not MoveKind.ACTIVATE.spends_acquisition and MoveKind.ADD_DROP.spends_acquisition


def test_made_means_the_drop_is_gone_and_the_player_stays():
    before = team(CENTRE, STREAMER, DEFENCE, GOALIE, BACK)
    after = team(CENTRE, DEFENCE, GOALIE, BACK)
    assert not MoveKind.ACTIVATE.made_on(before, "back", "streamer")
    assert MoveKind.ACTIVATE.made_on(after, "back", "streamer")
    released = team(CENTRE, STREAMER, DEFENCE, GOALIE)
    assert MoveKind.ACTIVATE.made_on(released, "back", "back")


def test_a_returning_player_whose_game_has_begun_comes_back_tomorrow():
    """A move Yahoo refuses fails the whole lineup: one made after a drop the same
    evening, for a player whose game had started, would have."""
    from dataclasses import replace

    locked = replace(BACK, is_editable=False)
    moves, alerts = legality.ir_changes(SMALL, team(CENTRE, DEFENCE, GOALIE, locked))
    assert moves == [] and "comes back tomorrow morning" in alerts[0]
    moves, _ = legality.ir_changes(SMALL, team(CENTRE, DEFENCE, GOALIE, BACK))
    assert [m.describe() for m in moves] == ["ACTIVATE Back IR+ -> BN"]
