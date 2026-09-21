"""The free-agent and waiver pool, and the timing question it answers."""

from __future__ import annotations

from puckpilot.season.matchups import for_date, for_week, parse_matchups, weeks_of
from puckpilot.season.pool import FREE_AGENT, WAIVERS, PoolPlayer, parse_player, rising


def entry(
    key="999.p.1",
    name="Test Player",
    team="LA",
    primary="C",
    eligible=("C", "Util"),
    owned=40,
    delta="3",
    status=None,
):
    core = [
        {"player_key": key},
        {"name": {"full": name}},
        {"editorial_team_abbr": team},
        {"primary_position": primary},
        {"eligible_positions": [{"position": p} for p in eligible]},
    ]
    if status:
        core += [{"status": status}]
    po = [{"coverage_type": "week", "week": 2}, {"value": owned}, {"delta": delta}]
    tail = {"percent_owned": po}
    return [core, tail]


# -- the abbreviation bug ---------------------------------------------------


def test_yahoo_club_abbreviations_are_normalised_to_the_nhl_ones():
    """Yahoo says LA, SJ, NJ and TB where the NHL schedule says LAK, SJS, NJD
    and TBL. This field is compared against the schedule, so leaving it alone
    silently reads as "has no game today" for four clubs' worth of players."""
    for yahoo, nhl in (("LA", "LAK"), ("SJ", "SJS"), ("NJ", "NJD"), ("TB", "TBL")):
        assert parse_player(entry(team=yahoo)).team == nhl


def test_a_club_that_already_matches_is_left_alone():
    assert parse_player(entry(team="TOR")).team == "TOR"


# -- ownership and its trend ------------------------------------------------


def test_percent_owned_and_its_delta_are_read():
    p = parse_player(entry(owned=57, delta="12"))
    assert p.percent_owned == 57.0
    assert p.percent_owned_delta == 12.0


def test_a_missing_delta_is_zero_not_a_crash():
    e = entry()
    e[1] = {"percent_owned": [{"value": 20}]}
    assert parse_player(e).percent_owned_delta == 0.0


def test_ownership_type_is_stamped_from_the_query():
    """`status=FA` cannot come back holding anything else, so asking Yahoo for
    the ownership subresource as well would spend a request to learn what we
    already said."""
    assert parse_player(entry(), ownership_type=FREE_AGENT).is_free_agent
    assert parse_player(entry(), ownership_type=WAIVERS).on_waivers


def test_rising_ranks_by_the_delta_and_skips_the_injured():
    a = parse_player(entry(key="p.1", name="Hot", delta="20"), ownership_type=FREE_AGENT)
    b = parse_player(entry(key="p.2", name="Warm", delta="5"), ownership_type=FREE_AGENT)
    c = parse_player(entry(key="p.3", name="Flat", delta="0"), ownership_type=FREE_AGENT)
    hurt = parse_player(
        entry(key="p.4", name="Hurt", delta="30", status="O"), ownership_type=FREE_AGENT
    )
    assert [p.name for p in rising([c, b, a, hurt])] == ["Hot", "Warm"]


# -- the 2am question -------------------------------------------------------


def test_a_waiver_claim_does_not_need_you_awake():
    p = PoolPlayer(
        player_key="p.1",
        name="X",
        team="TOR",
        primary_position="C",
        yahoo_eligible=frozenset({"C"}),
        ownership_type=WAIVERS,
    )
    assert "go to bed" in p.timing(waiver_days=1)


def test_a_free_agent_is_a_race():
    p = PoolPlayer(
        player_key="p.1",
        name="X",
        team="TOR",
        primary_position="C",
        yahoo_eligible=frozenset({"C"}),
        ownership_type=FREE_AGENT,
    )
    assert "race" in p.timing()


# -- matchups ---------------------------------------------------------------


def matchup_payload(*specs):
    node = {"count": len(specs)}
    for i, (week, start, end, them) in enumerate(specs):
        node[str(i)] = {
            "matchup": {
                "week": str(week),
                "week_start": start,
                "week_end": end,
                "status": "preevent",
                "is_playoffs": "0",
                "0": {
                    "teams": {
                        "count": 2,
                        "0": {"team": [[{"team_key": "999.l.1.t.5"}, {"name": "Us"}]]},
                        "1": {"team": [[{"team_key": them[0]}, {"name": them[1]}]]},
                    }
                },
            }
        }
    return {"fantasy_content": {"team": [[{"team_key": "999.l.1.t.5"}], {"matchups": node}]}}


def test_the_opponent_is_the_team_that_is_not_us():
    ms = parse_matchups(
        matchup_payload((1, "2026-09-29", "2026-10-04", ("999.l.1.t.11", "Them"))),
        our_team_key="999.l.1.t.5",
    )
    assert ms[0].opponent_name == "Them"
    assert ms[0].opponent_key == "999.l.1.t.11"


def test_week_boundaries_come_through_unmodified_including_a_long_week():
    """The real season has a fourteen-day week; computing weeks would put every
    later one seven days out."""
    ms = parse_matchups(
        matchup_payload(
            (1, "2026-09-29", "2026-10-04", ("t.2", "A")),
            (19, "2027-02-01", "2027-02-14", ("t.3", "B")),
        ),
        our_team_key="999.l.1.t.5",
    )
    weeks = weeks_of(ms)
    assert len(weeks[0].dates()) == 6
    assert len(weeks[1].dates()) == 14


def test_lookup_by_week_and_by_date():
    ms = parse_matchups(
        matchup_payload(
            (1, "2026-09-29", "2026-10-04", ("t.2", "A")),
            (2, "2026-10-05", "2026-10-11", ("t.3", "B")),
        ),
        our_team_key="999.l.1.t.5",
    )
    assert for_week(ms, 2).opponent_name == "B"
    assert for_date(ms, "2026-10-07").week == 2
    assert for_date(ms, "2027-01-01") is None


def test_a_matchup_without_dates_is_skipped_not_fatal():
    bad = matchup_payload((1, "", "", ("t.2", "A")))
    assert parse_matchups(bad, our_team_key="999.l.1.t.5") == []


def test_a_payload_that_is_not_a_team_gives_nothing():
    assert parse_matchups({"fantasy_content": {"league": []}}) == []
