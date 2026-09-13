"""Yahoo player_key -> NHL player_id matching.

Every live draft pick crosses this bridge, and the failure modes are asymmetric:
a MISSED match leaves a drafted player on our board (bad), while a WRONG match
removes someone else (worse, and invisible). So the fallbacks are tested for
what they refuse as much as what they accept.

The awkward cases here are all real, taken from a 2026-27 league pool.
"""

from __future__ import annotations

import pytest

from puckpilot.data import store
from puckpilot.yahoo.playermap import _nhl_index, _resolve, _team, reresolve_unmatched


@pytest.fixture
def idx(db):
    # Names on the left are exactly as our DB holds them, including the two
    # where MoneyPuck dropped an accented letter instead of folding it.
    for pid, name, team in [
        (1, "Connor McDavid", "EDM"),
        (2, "Mitchell Marner", "TOR"),  # Yahoo says "Mitch Marner", and VGK
        (3, "Tim Sttzle", "OTT"),  # really Stützle
        (4, "Alexis Lafrenire", "NYR"),  # really Lafrenière
        (5, "Yegor Chinakhov", "CBJ"),  # Yahoo transliterates "Egor"
        (6, "Nicholas Paul", "TBL"),
        (7, "Sebastian Aho", "CAR"),  # two players share this name
        (8, "Sebastian Aho", "NYI"),
        (9, "Matt Murray", "OTT"),
        (10, "Matt Murray", "SEA"),
    ]:
        store.upsert_player(db, pid, name, "C", team)
    return _nhl_index(db)


def test_exact_name_matches(idx):
    assert _resolve(idx, "Connor McDavid", "EDM") == (1, "exact")


def test_duplicate_names_are_split_by_team(idx):
    assert _resolve(idx, "Sebastian Aho", "CAR") == (7, "exact+team")
    assert _resolve(idx, "Sebastian Aho", "NYI") == (8, "exact+team")


def test_duplicate_name_on_an_unknown_team_is_refused(idx):
    """Two Matt Murrays and neither on the team Yahoo reports: refuse."""
    pid, how = _resolve(idx, "Matt Murray", "VGK")
    assert pid is None and how == "ambiguous"


def test_nickname_matches_despite_a_stale_team(idx):
    """`nhl_players.team_abbrev` is the LAST-PLAYED team, so a traded player
    disagrees with Yahoo. Marner reads TOR here and VGK on Yahoo; the match has
    to survive that or a genuine star stays on the board all draft."""
    pid, how = _resolve(idx, "Mitch Marner", "VGK")
    assert pid == 2 and how.startswith("surname-unique")


def test_transliteration_variant_matches(idx):
    pid, how = _resolve(idx, "Egor Chinakhov", "CBJ")
    assert pid == 5 and how.startswith(("surname", "exact"))


def test_given_name_variant_matches(idx):
    pid, _ = _resolve(idx, "Nick Paul", "TBL")
    assert pid == 6


def test_dropped_accent_matches_on_the_same_team(idx):
    """MoneyPuck supplies 'Tim Sttzle'; Yahoo spells it correctly. Exact and
    surname matching both fail, so the fuzzy layer has to carry it."""
    pid, how = _resolve(idx, "Tim Stutzle", "OTT")
    assert pid == 3 and how.startswith("fuzzy")
    pid2, _ = _resolve(idx, "Alexis Lafreniere", "NYR")
    assert pid2 == 4


def test_unrelated_name_is_never_force_matched(idx):
    """A prospect who simply is not in our NHL data must come back unmatched,
    not attached to whoever happens to be closest."""
    pid, how = _resolve(idx, "Gavin McKenna", "TOR")
    assert pid is None and how == "unmatched"


def test_shared_surname_alone_does_not_match(idx):
    """'Aho' is not unique, so a wrong given name plus an unknown team must not
    fall through to a surname match."""
    pid, _ = _resolve(idx, "Wilhelm Aho", "DAL")
    assert pid is None


def test_team_aliases_normalize_yahoo_abbreviations():
    assert _team("LA") == "LAK"
    assert _team("SJ") == "SJS"
    assert _team("TB") == "TBL"
    assert _team("EDM") == "EDM"
    assert _team(None) == ""


# ---- re-resolving previously-unmatched rows, without asking Yahoo again ----


def _map_row(db, key, name, team, nhl_id=None, adp=100):
    db.execute(
        "INSERT INTO yahoo_player_map"
        " (player_key, league_key, full_name, team_abbrev, positions, nhl_player_id, adp_rank)"
        " VALUES (?, 'L.1', ?, ?, 'C', ?, ?)",
        (key, name, team, nhl_id, adp),
    )


def test_a_rookie_synced_after_the_fact_resolves_on_reresolve(db):
    """The exact bug this exists to fix: McKenna failed to match when the
    Yahoo pool was first fetched because `nhl_players` had no row for him yet.
    A roster sync since then gave him one; re-resolving must pick it up
    without going back to Yahoo."""
    _map_row(db, "477.p.1", "Gavin McKenna", "TOR")
    db.commit()

    before = reresolve_unmatched(db)
    assert before.matched == 0 and before.unmatched == ["Gavin McKenna"]

    store.upsert_player(db, 999, "Gavin McKenna", "L", "TOR")
    after = reresolve_unmatched(db)
    assert after.matched == 1
    row = db.execute(
        "SELECT nhl_player_id FROM yahoo_player_map WHERE player_key = '477.p.1'"
    ).fetchone()
    assert row[0] == 999


def test_reresolve_never_touches_adp_or_positions(db):
    """Only `nhl_player_id` may change here - `adp_rank` and `positions` were
    not re-fetched, so nothing here may act as if they were."""
    _map_row(db, "477.p.1", "Connor McDavid", "EDM", adp=1)
    store.upsert_player(db, 1, "Connor McDavid", "C", "EDM")

    reresolve_unmatched(db)

    row = db.execute(
        "SELECT positions, adp_rank FROM yahoo_player_map WHERE player_key = '477.p.1'"
    ).fetchone()
    assert tuple(row) == ("C", 1)


def test_reresolve_does_not_double_assign_an_nhl_id(db):
    """An id another Yahoo row already claims - in this league or any other -
    must not be handed out a second time."""
    _map_row(db, "477.p.1", "Sebastian Aho", "CAR", nhl_id=7)
    _map_row(db, "477.p.2", "Sebastian Aho", "CAR")  # unresolved twin, same name+team
    store.upsert_player(db, 7, "Sebastian Aho", "C", "CAR")

    report = reresolve_unmatched(db)

    assert report.matched == 0
    row = db.execute(
        "SELECT nhl_player_id FROM yahoo_player_map WHERE player_key = '477.p.2'"
    ).fetchone()
    assert row[0] is None
