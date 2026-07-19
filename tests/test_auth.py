import pytest

from puckpilot.config import Settings
from puckpilot.yahoo.auth import MissingYahooCredentials, extract_code, get_oauth_session


def test_extract_code_from_full_redirect_url():
    url = "https://localhost:9000/?code=q6gtabc123"
    assert extract_code(url) == "q6gtabc123"
    assert extract_code("https://localhost:9000/?foo=1&code=xyz&bar=2") == "xyz"


def test_extract_code_from_bare_code():
    assert extract_code("  abc123x \n") == "abc123x"


def _bare_settings(**overrides) -> Settings:
    # _env_file=None: ignore any real .env on this machine
    return Settings(_env_file=None, **overrides)


def test_missing_credentials_raise_with_setup_help(monkeypatch):
    monkeypatch.delenv("YAHOO_CLIENT_ID", raising=False)
    monkeypatch.delenv("YAHOO_CLIENT_SECRET", raising=False)
    with pytest.raises(MissingYahooCredentials, match="developer.yahoo.com"):
        get_oauth_session(_bare_settings())


def test_partial_credentials_also_raise(monkeypatch):
    monkeypatch.delenv("YAHOO_CLIENT_SECRET", raising=False)
    with pytest.raises(MissingYahooCredentials):
        get_oauth_session(_bare_settings(yahoo_client_id="abc"))
