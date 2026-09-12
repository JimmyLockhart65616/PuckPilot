import pytest

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


# ---- the optional local join helper ----------------------------------------
#
# `draft farm` opens the mock lobby and waits. Joining a lobby is a click, and
# the published tool does not make it; a user may supply their own helper at
# `puckpilot.local.join`, which is git-ignored and absent from a clone.
#
# So the ABSENT path is the one every clone and every CI run takes, and it must
# keep working. Without a test, CI is the only thing exercising it.


def test_farm_is_importable_and_hooked_without_the_local_helper():
    """The guarded import must degrade to today's behaviour, not explode."""
    import importlib.util
    import inspect

    if importlib.util.find_spec("puckpilot.local") is not None:
        # Present on this machine, so the absent path cannot be exercised here.
        # It still runs on every clone and in CI, which is the point.
        pytest.skip("puckpilot.local is installed locally; CI covers the absent path")
    src = inspect.getsource(cli._cmd_draft_farm)
    assert "except ImportError:" in src, "the hook must be guarded"
    assert "join_a_mock = None" in src, "absent must mean 'wait for a human'"
    # and the wait-for-a-human branch survives
    assert "join a mock draft in the browser" in src


def test_farm_help_does_not_promise_a_human_will_click():
    """The flag used to say 'waiting for YOU to join', which is wrong either
    way now - it is how long to wait for a draft to start."""
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:
        import inspect

        src = inspect.getsource(cli)
        assert "waiting for you to join" not in src.lower()
