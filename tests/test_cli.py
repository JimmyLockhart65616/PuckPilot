from puckpilot import cli
from puckpilot.yahoo.auth import MissingYahooCredentials


def test_league_show_without_creds_exits_2_with_help(monkeypatch, capsys):
    def boom(settings):
        raise MissingYahooCredentials()

    monkeypatch.setattr("puckpilot.yahoo.auth.get_oauth_session", boom)
    rc = cli.main(["league", "show"])
    assert rc == 2
    assert "developer.yahoo.com" in capsys.readouterr().err


def test_data_init_creates_db_at_env_path(monkeypatch, tmp_path, capsys):
    db = tmp_path / "cli.db"
    monkeypatch.setenv("DB_PATH", str(db))
    rc = cli.main(["data", "init"])
    assert rc == 0
    assert db.exists()
    out = capsys.readouterr().out
    assert "nhl_game_logs" in out
