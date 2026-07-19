import pytest
import respx

from puckpilot.data.moneypuck import BASE_URL, MoneyPuckClient, season_start_year

CSV = (
    "playerId,season,name,team,position,situation,games_played,icetime\n"
    "8478402,2025,Connor McDavid,EDM,C,all,82,120000\n"
    "8478402,2025,Connor McDavid,EDM,C,5on4,82,20000\n"
)


def test_season_start_year():
    assert season_start_year("20252026") == 2025
    for bad in ("2025", "20252027", "abcdefgh"):
        with pytest.raises(ValueError):
            season_start_year(bad)


@respx.mock
def test_season_csv_downloads_parses_and_caches(tmp_path):
    route = respx.get(f"{BASE_URL}/2025/regular/skaters.csv").respond(content=CSV.encode())
    mp = MoneyPuckClient(cache_dir=tmp_path)
    rows = mp.season_csv("20252026", "skaters")
    assert rows is not None
    assert rows[0]["playerId"] == "8478402"
    assert rows[0]["situation"] == "all"
    assert rows[1]["situation"] == "5on4"
    assert (tmp_path / "20252026_skaters.csv").exists()
    # second call is served from the file cache, no network hit
    assert mp.season_csv("20252026", "skaters") == rows
    assert route.call_count == 1
    # refresh forces a re-download
    mp.season_csv("20252026", "skaters", refresh=True)
    assert route.call_count == 2


@respx.mock
def test_season_csv_404_returns_none(tmp_path):
    respx.get(f"{BASE_URL}/2026/regular/skaters.csv").respond(404)
    assert MoneyPuckClient(cache_dir=tmp_path).season_csv("20262027", "skaters") is None


def test_bad_kind_raises(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        MoneyPuckClient(cache_dir=tmp_path).season_csv("20252026", "forwards")
