"""Phone notifications: what is sent, and that nothing is sent or broken without a URL."""

from __future__ import annotations

from types import SimpleNamespace

from puckpilot.season import notify


class _Post:
    def __init__(self, status=200, boom=False):
        self.calls = []
        self.status = status
        self.boom = boom

    def __call__(self, url, content, headers, timeout):
        if self.boom:
            raise OSError("network down")
        self.calls.append((url, content.decode("utf-8"), headers))
        return SimpleNamespace(status_code=self.status)


def _proposal(add, drop="", gain=None, helps=()):
    return SimpleNamespace(
        add_name=add,
        drop_name=drop,
        drop_player_key="k" if drop else "",
        reason={"expected_gain": gain, "helps": list(helps)},
    )


def test_nothing_is_sent_without_a_url(monkeypatch):
    monkeypatch.delenv(notify.NOTIFY_ENV, raising=False)
    post = _Post()
    assert notify.send("t", "b", post=post) is False
    assert post.calls == []


def test_a_new_pickup_says_who_for_whom_and_what_it_is_worth(monkeypatch):
    monkeypatch.setenv(notify.NOTIFY_ENV, "https://ntfy.example/topic")
    post = _Post()
    sent = notify.new_pickups(
        [_proposal("Pickup", "Depth", 0.42, ("HIT", "BLK", "SOG"))], "https://page", post=post
    )
    assert sent
    url, body, headers = post.calls[0]
    assert url == "https://ntfy.example/topic"
    assert body == "Add Pickup for Depth: +0.42 categories this week (HIT, BLK)"
    assert headers["Title"] == "New pickup to decide on"
    assert headers["Click"] == "https://page"  # the bare page; never its key


def test_several_pickups_are_one_notification(monkeypatch):
    monkeypatch.setenv(notify.NOTIFY_ENV, "https://ntfy.example/topic")
    post = _Post()
    notify.new_pickups([_proposal("A"), _proposal("B", "C")], post=post)
    assert len(post.calls) == 1
    assert post.calls[0][2]["Title"] == "2 pickups to decide on"
    assert post.calls[0][1].splitlines() == ["Add A", "Add B for C"]


def test_a_failure_is_urgent_and_a_dead_service_never_raises(monkeypatch):
    monkeypatch.setenv(notify.NOTIFY_ENV, "https://ntfy.example/topic")
    post = _Post()
    assert notify.failed("tonight's lineup", "a dialog was in the way", post=post)
    assert post.calls[0][2]["Priority"] == "high"
    assert post.calls[0][2]["Title"] == "NOT done: tonight's lineup"
    assert notify.failed("x", "y", post=_Post(boom=True)) is False
    assert notify.send("t", "b", post=_Post(status=500)) is False


def test_a_title_with_any_character_still_sends(monkeypatch):
    """HTTP headers are latin-1: a dash or an accent must not break a send."""
    monkeypatch.setenv(notify.NOTIFY_ENV, "https://ntfy.example/topic")
    post = _Post()
    assert notify.send("Stützle – out", "b", post=post)
    assert post.calls[0][2]["Title"].startswith("St")


def test_a_run_notifies_its_new_pickups_and_a_missed_lineup(db, monkeypatch):
    from puckpilot.season import proposals as proposals_mod
    from puckpilot.season.run import RunReport, _notify

    sent = []
    monkeypatch.setattr(notify, "new_pickups", lambda ps, url="": sent.append(("new", ps)) or True)
    monkeypatch.setattr(
        notify, "failed", lambda what, why, url="": sent.append(("fail", why)) or True
    )
    db.execute(
        "INSERT INTO waiver_proposals (created_at, add_pid, drop_pid, reason_json, status, manager,"
        " league_key, team_key, kind, add_player_key, drop_player_key) VALUES"
        " ('2026-10-06 11:00:00', 1, NULL, '{}', 'pending', 'm', 'L', 'T', 'add', 'a1', NULL),"
        " ('2026-10-06 09:00:00', 2, NULL, '{}', 'pending', 'm', 'L', 'T', 'add', 'a2', NULL)"
    )
    db.commit()
    manager = SimpleNamespace(name="m", page=SimpleNamespace(url="https://page", publishes=True))
    report = RunReport(date="2026-10-06", manager="m")
    report.add("act", False, "TimeoutError: a dialog was in the way")
    _notify(db, manager, "L", report, since="2026-10-06 10:00:00")
    assert [k for k, _ in sent] == ["new", "fail"]
    assert [p.add_player_key for p in sent[0][1]] == ["a1"]  # only what this run queued
    assert report.steps[-1].name == "notify"
    sent.clear()
    quiet = RunReport(date="2026-10-06", manager="m")
    quiet.add("act", False, "switched off (PUCKPILOT_ACT_OFF is set) - nothing changed")
    _notify(db, manager, "L", quiet, since="2026-10-06 12:00:00")
    assert sent == []  # the kill switch is not news
    assert proposals_mod.PENDING == "pending"


def test_an_approval_asked_again_says_why(monkeypatch):
    from puckpilot.season import notify

    got = []
    monkeypatch.setenv(notify.NOTIFY_ENV, "https://ntfy.example/topic")
    notify.ask_again("#12 add Pickup", "https://page", post=lambda url, **k: got.append(k) or _Ok())
    assert got[0]["headers"]["Title"] == "One more tap: #12 add Pickup"
    assert "make it yourself" in got[0]["content"].decode()


class _Ok:
    status_code = 200
