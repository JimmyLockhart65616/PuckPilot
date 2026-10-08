"""Loading the manager the tool is acting for."""

from __future__ import annotations

from pathlib import Path

import pytest

from puckpilot.season.manager import Manager, ManagerError, available, load_manager

TEMPLATE = Path(__file__).resolve().parents[1] / "managers" / "example.toml"


class _Settings:
    """Settings with the repo root pointed at a tmp dir."""

    def __init__(self, root: Path):
        self.root = root

    def _resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    @property
    def resolved_db_path(self) -> Path:
        return self.root / "data" / "puckpilot.db"


@pytest.fixture
def home(tmp_path):
    (tmp_path / "managers").mkdir()
    return tmp_path


def write(home, name, body):
    (home / "managers" / f"{name}.toml").write_text(body, encoding="utf-8")
    return _Settings(home)


def test_the_shipped_template_parses():
    """A template that does not load is worse than none."""
    root = TEMPLATE.resolve().parents[1]
    m = load_manager("example", _Settings(root))
    assert m.name == "example"
    assert m.authority.lineup.enabled is False
    assert m.authority.transactions.requires_approval is True


def test_a_missing_manager_says_how_to_make_one(home):
    with pytest.raises(ManagerError, match="Copy managers/example.toml"):
        load_manager("nobody", _Settings(home))


def test_a_manager_without_a_profile_can_decide_but_not_act(home):
    s = write(home, "view", 'name = "view"\n')
    m = load_manager("view", s)
    assert m.can_act is False


def test_a_manager_with_a_profile_can_act(home):
    s = write(home, "act", 'name = "act"\nprofile_dir = "secrets/p"\n')
    m = load_manager("act", s)
    assert m.can_act is True
    assert m.profile_dir == home / "secrets" / "p"


def test_two_managers_are_independent(home):
    """Same league, different chairs, and they are rivals: no shared rows."""
    write(home, "one", 'name = "one"\nteam_key = "1.l.1.t.3"\ndb_path = "a.db"\n')
    s = write(home, "two", 'name = "two"\nteam_key = "1.l.1.t.9"\ndb_path = "b.db"\n')
    a, b = load_manager("one", s), load_manager("two", s)
    assert a.team_key != b.team_key
    assert a.resolved_db() != b.resolved_db()


def test_page_keys_are_per_manager(home):
    s = write(home, "p", 'name = "p"\n[page]\nurl = "https://x"\nguest_key = "g"\n')
    m = load_manager("p", s)
    assert m.page.publishes is True
    assert m.page.guest_key == "g"


def test_no_page_configured_means_no_publish(home):
    m = load_manager("q", write(home, "q", 'name = "q"\n'))
    assert m.page.publishes is False


def test_invalid_toml_names_the_file(home):
    s = write(home, "bad", "name = [[[")
    with pytest.raises(ManagerError, match="invalid TOML"):
        load_manager("bad", s)


def test_a_bad_authority_block_names_the_file(home):
    s = write(home, "bad2", 'name = "b"\n[authority.transactions]\nenabled = true\n')
    with pytest.raises(ManagerError, match="not a setting"):
        load_manager("bad2", s)


def test_available_lists_configs_but_not_the_template(home):
    write(home, "example", 'name = "example"\n')
    s = write(home, "real", 'name = "real"\n')
    assert available(s) == ["real"]


def test_describe_never_prints_a_key(home):
    s = write(home, "k", 'name = "k"\n[page]\nurl = "https://x"\nowner_key = "SECRET"\n')
    assert "SECRET" not in load_manager("k", s).describe()


def test_a_bare_manager_falls_back_to_the_default_database():
    m = Manager(name="x")
    assert m.resolved_db().name.endswith(".db")


# -- the browser dying under a read -------------------------------------------


class TargetClosedError(Exception):
    """Stands in for playwright's, which run_session recognises by name."""


def _sessions(monkeypatch, failures):
    import contextlib

    from puckpilot.season import cli_support

    opened = []

    @contextlib.contextmanager
    def fake_open(manager, settings=None):
        opened.append(1)
        if len(opened) <= failures:
            raise TargetClosedError("Target page, context or browser has been closed")
        yield "session"

    monkeypatch.setattr(cli_support, "open_session", fake_open)
    return cli_support, opened


def test_a_browser_that_dies_under_a_read_is_retried_once(monkeypatch):
    cli_support, opened = _sessions(monkeypatch, failures=1)
    waited = []
    got = cli_support.run_session(None, lambda s: f"read with {s}", sleep=waited.append)
    assert got == "read with session" and len(opened) == 2
    assert waited == [cli_support.BROWSER_RETRY_WAIT_S]


def test_it_is_retried_only_once(monkeypatch):
    import pytest

    cli_support, opened = _sessions(monkeypatch, failures=2)
    with pytest.raises(TargetClosedError):
        cli_support.run_session(None, lambda s: s, sleep=lambda _s: None)
    assert len(opened) == 2


def test_any_other_failure_is_not_retried(monkeypatch):
    import contextlib

    import pytest

    from puckpilot.season import cli_support

    @contextlib.contextmanager
    def broken(manager, settings=None):
        raise ValueError("not a browser problem")
        yield  # pragma: no cover

    monkeypatch.setattr(cli_support, "open_session", broken)
    with pytest.raises(ValueError):
        cli_support.run_session(None, lambda s: s, sleep=lambda _s: pytest.fail("retried"))


def test_a_request_the_page_dropped_is_retried_once(monkeypatch):
    """2026-10-07 19:10: "Failed to fetch" on one read, and the same read a
    minute later worked. A read is safe to run again."""
    import contextlib

    from puckpilot.season import cli_support

    @contextlib.contextmanager
    def fake_open(manager, settings=None):
        yield "session"

    monkeypatch.setattr(cli_support, "open_session", fake_open)
    tries, waited = [], []

    def work(s):
        tries.append(s)
        if len(tries) == 1:
            raise RuntimeError("Page.evaluate: TypeError: Failed to fetch")
        return "read"

    assert cli_support.run_session(None, work, sleep=waited.append) == "read"
    assert len(tries) == 2 and waited == [cli_support.FETCH_RETRY_WAIT_S]
