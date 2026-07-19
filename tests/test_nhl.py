import pytest
import respx

from puckpilot.data.nhl import BASE_URL, NhlApiError, NhlClient


@respx.mock
def test_player_landing():
    respx.get(f"{BASE_URL}/player/8478402/landing").respond(
        json={"playerId": 8478402, "lastName": {"default": "McDavid"}, "position": "C"}
    )
    data = NhlClient().player_landing(8478402)
    assert data["lastName"]["default"] == "McDavid"


@respx.mock
def test_game_log_path_includes_season_and_type():
    route = respx.get(f"{BASE_URL}/player/8478402/game-log/20252026/2").respond(
        json={"gameLog": [{"goals": 1, "assists": 2, "points": 3}]}
    )
    data = NhlClient().player_game_log(8478402, "20252026")
    assert route.called
    assert data["gameLog"][0]["points"] == 3


@respx.mock
def test_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setattr("puckpilot.data.nhl.time.sleep", lambda s: None)
    route = respx.get(f"{BASE_URL}/standings/now")
    route.side_effect = [
        respx.MockResponse(503),
        respx.MockResponse(200, json={"standings": []}),
    ]
    data = NhlClient().standings_now()
    assert data == {"standings": []}
    assert route.call_count == 2


@respx.mock
def test_no_retry_on_404(monkeypatch):
    monkeypatch.setattr("puckpilot.data.nhl.time.sleep", lambda s: None)
    route = respx.get(f"{BASE_URL}/player/1/landing").respond(404)
    with pytest.raises(NhlApiError, match="HTTP 404"):
        NhlClient().player_landing(1)
    assert route.call_count == 1
