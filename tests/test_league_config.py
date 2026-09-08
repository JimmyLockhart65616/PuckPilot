import pytest

from puckpilot.engine.categories import UnknownCategory, resolve
from puckpilot.league import LeagueConfig, LeagueConfigError, load_league

MINIMAL = """
name = "Test League"
[roster]
teams = 8
slots = [{ pos = "C", count = 1 }, { pos = "G", count = 1 }]
util = 1
bench = 2
[scoring]
type = "roto"
skater = ["G", "A"]
goalie = ["W", "SV%"]
"""


def _write(tmp_path, text, name="l.toml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_league_reads_shape_and_categories(tmp_path):
    lg = load_league(_write(tmp_path, MINIMAL))
    assert lg.name == "Test League"
    assert lg.scoring == "roto"
    assert lg.shape.n_teams == 8
    assert lg.shape.slots == (("C", 1), ("G", 1))
    assert lg.shape.roster_size == 5  # 1 C + 1 G + 1 util + 2 bench
    assert [c.label for c in lg.skater_cats] == ["G", "A"]
    assert [c.label for c in lg.goalie_cats] == ["W", "SV%"]
    # no keepers configured -> every roster spot is drafted
    assert lg.n_keepers == 0
    assert lg.draft_rounds == 5


def test_keepers_reduce_draft_rounds(tmp_path):
    lg = load_league(
        _write(
            tmp_path,
            MINIMAL
            + """
[keepers]
count = 2
[keepers.by_season]
"20262027" = ["Connor McDavid", "Cale Makar"]
""",
        )
    )
    assert lg.n_keepers == 2
    assert lg.draft_rounds == 3  # 5 roster spots - 2 keepers
    assert lg.keepers_for_season("20262027") == ("Connor McDavid", "Cale Makar")
    assert lg.keepers_for_season("20252026") == ()  # unlisted season -> no keepers


def test_shipped_example_config_is_valid():
    """The template others copy must actually parse."""
    lg = load_league("leagues/example.toml")
    assert lg.shape.n_teams == 12
    assert lg.min_goalie_appearances == 3
    assert lg.season_acquisitions == 65


def test_unknown_category_names_the_valid_ones(tmp_path):
    bad = MINIMAL.replace('skater = ["G", "A"]', 'skater = ["G", "TOTALLY_FAKE"]')
    with pytest.raises(UnknownCategory) as e:
        load_league(_write(tmp_path, bad))
    assert "TOTALLY_FAKE" in str(e.value)
    assert "HIT" in str(e.value)  # lists what IS available


def test_bad_scoring_type_is_rejected(tmp_path):
    bad = MINIMAL.replace('type = "roto"', 'type = "points"')
    with pytest.raises(LeagueConfigError, match="h2h"):
        load_league(_write(tmp_path, bad))


def test_malformed_slots_explain_the_expected_shape(tmp_path):
    bad = MINIMAL.replace(
        'slots = [{ pos = "C", count = 1 }, { pos = "G", count = 1 }]', 'slots = ["C", "G"]'
    )
    with pytest.raises(LeagueConfigError, match="pos"):
        load_league(_write(tmp_path, bad))


def test_missing_file_raises_clearly(tmp_path):
    with pytest.raises(LeagueConfigError, match="not found"):
        load_league(tmp_path / "nope.toml")


def test_invalid_toml_names_the_file(tmp_path):
    with pytest.raises(LeagueConfigError, match="invalid TOML"):
        load_league(_write(tmp_path, "name = = broken"))


def test_category_labels_resolve_case_insensitively():
    assert resolve("sv%").key == "save_pct"
    assert resolve("HIT").key == "hits"
    assert resolve("GAA").higher_is_better is False


def test_default_config_needs_no_file():
    """A fresh clone with no league file still gets a usable 12-team default."""
    lg = LeagueConfig()
    assert lg.shape.n_teams == 12
    assert lg.all_cats
