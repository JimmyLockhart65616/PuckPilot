"""Parsing a live roster, including the two traps that make `flatten` unsafe here.

Payload shapes mirror a real `/team/{key}/roster` response; values are invented.
Assertions against the actual roster live in `test_local_data.py`.
"""

from __future__ import annotations

import pytest

from puckpilot.season.roster import RosterError, parse_player, parse_roster


def player(
    key="999.p.1",
    name="Test Player",
    team="TOR",
    primary="C",
    eligible=("C", "Util"),
    slot="C",
    status=None,
    editable=1,
    undroppable="0",
    keeper=False,
    add_positions=("LW", "D"),
):
    """A player entry in Yahoo's real shape.

    Note what is deliberately present: `is_keeper.status` and
    `eligible_positions_to_add[].position` both collide with the player's own
    `status` and selected `position` under any recursive merge.
    """
    core = [
        {"player_key": key},
        {"player_id": key.rsplit(".", 1)[-1]},
        {"name": {"full": name, "first": name.split()[0], "last": name.split()[-1]}},
    ]
    if status:
        core += [{"status": status[0]}, {"status_full": status[1]}, {"injury_note": status[2]}]
    core += [
        {"editorial_team_abbr": team},
        {"is_keeper": {"status": keeper, "cost": False, "kept": keeper}},
        {"is_undroppable": undroppable},
        {"primary_position": primary},
        {"eligible_positions": [{"position": p} for p in eligible]},
        {"eligible_positions_to_add": [{"position": p} for p in add_positions]},
    ]
    sel = [{"coverage_type": "date", "date": "2026-10-07"}, {"position": slot}]
    return [core, {"selected_position": sel}, {"is_editable": editable}]


def roster_payload(entries, date="2026-10-07", editable=1):
    players = {"count": len(entries)}
    for i, e in enumerate(entries):
        players[str(i)] = {"player": e}
    return {
        "fantasy_content": {
            "team": [
                [{"team_key": "999.l.1.t.5"}, {"name": "Test Team"}],
                {
                    "roster": {
                        "coverage_type": "date",
                        "date": date,
                        "is_editable": editable,
                        "0": {"players": players},
                    }
                },
            ]
        }
    }


# -- the traps --------------------------------------------------------------


def test_a_keeper_is_not_reported_as_injured():
    """`is_keeper.status` is a bool on a nested object; a merge reads it as the
    player's injury status. On the real 2026-09-18 roster this marked all three
    keepers hurt."""
    p = parse_player(player(keeper=True))
    assert p.status == ""
    assert p.is_out is False


def test_the_selected_slot_is_not_read_off_eligible_positions_to_add():
    p = parse_player(player(primary="C", eligible=("C", "LW", "RW", "Util"), slot="Util"))
    assert p.selected_slot == "Util"
    assert p.eligible == {"C", "L", "R"}


# -- ordinary parsing -------------------------------------------------------


def test_status_is_carried_whole():
    p = parse_player(player(status=("O", "Out", "Knee")))
    assert (p.status, p.status_full, p.injury_note) == ("O", "Out", "Knee")
    assert p.is_out is True and p.is_questionable is False


def test_day_to_day_is_not_out():
    """DTD players play most nights; treating them as out benches half a roster."""
    p = parse_player(player(status=("DTD", "Day-to-Day", "Upper body")))
    assert p.is_out is False and p.is_questionable is True


def test_yahoo_wings_become_engine_positions_and_slots_are_dropped():
    p = parse_player(player(eligible=("C", "LW", "RW", "Util", "IR+")))
    assert p.eligible == {"C", "L", "R"}
    assert "Util" in p.yahoo_eligible and "IR+" in p.yahoo_eligible


def test_bench_ir_and_starting_are_distinguished():
    assert parse_player(player(slot="BN")).on_bench
    assert parse_player(player(slot="IR+")).on_ir
    assert parse_player(player(slot="C")).starting
    assert not parse_player(player(slot="BN")).starting


def test_a_locked_player_is_flagged():
    assert parse_player(player(editable=0)).is_editable is False


def test_undroppable_is_read():
    assert parse_player(player(undroppable="1")).is_undroppable is True


# -- the whole roster -------------------------------------------------------


def test_roster_splits_starters_bench_and_injured():
    r = parse_roster(
        roster_payload(
            [
                player(key="999.p.1", name="Starter One", slot="C"),
                player(key="999.p.2", name="Benched One", slot="BN"),
                player(key="999.p.3", name="Hurt One", slot="IR", status=("O", "Out", "Knee")),
            ]
        ),
        team_key="999.l.1.t.5",
    )
    assert len(r) == 3
    assert [p.name for p in r.starters()] == ["Starter One"]
    assert [p.name for p in r.bench()] == ["Benched One"]
    assert [p.name for p in r.injured()] == ["Hurt One"]
    assert r.date == "2026-10-07"


def test_unmapped_players_are_named_not_dropped():
    """A call-up Yahoo knows and our map does not must stay visible."""
    r = parse_roster(
        roster_payload([player(key="999.p.1", name="Known"), player(key="999.p.2", name="Rookie")]),
        team_key="999.l.1.t.5",
        player_map={"999.p.1": 8400001},
    )
    assert len(r) == 2
    assert r.unmapped == ("Rookie",)
    assert r.by_nhl_id() == {8400001: r.players[0]}


def test_find_matches_exactly_then_loosely():
    r = parse_roster(
        roster_payload([player(key="999.p.1", name="Mathew Barzal")]), team_key="999.l.1.t.5"
    )
    assert r.find("mathew barzal").name == "Mathew Barzal"
    assert r.find("barzal").name == "Mathew Barzal"
    assert r.find("999.p.1").name == "Mathew Barzal"
    assert r.find("nobody") is None


def test_an_empty_roster_is_not_an_error():
    r = parse_roster(roster_payload([]), team_key="999.l.1.t.5")
    assert len(r) == 0 and r.starters() == ()


def test_a_non_team_payload_is_refused():
    with pytest.raises(RosterError, match="not a Yahoo team response"):
        parse_roster({"fantasy_content": {"league": []}}, team_key="999.l.1.t.5")


def test_league_key_is_derived_from_the_team_key():
    r = parse_roster(roster_payload([]), team_key="999.l.1.t.5")
    assert r.league_key == "999.l.1"


# -- Yahoo's own counters ---------------------------------------------------


def _with_counters(payload, minimum=None, adds=None, moves=None):
    team = payload["fantasy_content"]["team"]
    if adds is not None:
        team[0].append({"roster_adds": {"coverage_type": "week", "coverage_value": 2, "value": adds}})
    if moves is not None:
        team[0].append({"number_of_moves": moves})
    if minimum is not None:
        team[1]["roster"]["minimum_games"] = {
            "coverage_type": "week",
            "coverage_value": "2",
            "value": minimum,
        }
    return payload


def test_the_goalie_count_and_adds_are_read_off_the_payload():
    """Yahoo's numbers are the authority: our snapshots record slots, not who
    played, and counting slots read the goalie minimum as met by Tuesday."""
    r = parse_roster(
        _with_counters(roster_payload([]), minimum="2", adds="1", moves=7),
        team_key="999.l.1.t.5",
    )
    assert (r.goalie_games, r.goalie_games_week) == (2, 2)
    assert (r.adds_this_week, r.adds_week) == (1, 2)
    assert r.moves_season == 7


def test_absent_counters_are_unknown_not_zero():
    """Zero is a claim - "no adds used" - and an absent field makes none."""
    r = parse_roster(roster_payload([]), team_key="999.l.1.t.5")
    assert r.goalie_games is None
    assert r.adds_this_week is None
    assert r.moves_season is None
