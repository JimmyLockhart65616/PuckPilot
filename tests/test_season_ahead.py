"""The rest of the week set ahead, so a missed run leaves a planned day."""

from __future__ import annotations

from types import SimpleNamespace

from puckpilot.season.authority import LineupAuthority
from puckpilot.season.run import RunReport, _set_ahead

WEEK = SimpleNamespace(
    start="2026-10-12",
    dates=lambda: [f"2026-10-{d}" for d in range(12, 19)],
)


def _manager(**lineup):
    lineup = {"enabled": True, **lineup}
    return SimpleNamespace(name="m", authority=SimpleNamespace(lineup=LineupAuthority(**lineup)))


def _roster(*keys):
    return SimpleNamespace(team_key="T", players=[SimpleNamespace(player_key=k) for k in keys])


class _Move:
    def describe(self):
        return "START A in C"


def _plan_for(changes_on):
    def plan_for(r, d):
        moves = [_Move()] if d in changes_on else []
        return SimpleNamespace(moves=moves, ir_moves=[], within_authority=True,
                               ir_within_authority=True, date=d)  # fmt: skip

    return plan_for


class _Apply:
    def __init__(self, ok=True):
        self.dates, self.ok = [], ok

    def __call__(self, manager, team_key, date, phases):
        self.dates.append(date)
        said = "made 1 change(s)" if self.ok else "no"
        return SimpleNamespace(ok=self.ok, message=said, lines=[])


def _go(db, day="2026-10-15", roster=None, manager=None, apply=None, moved=False,
        changes_on=("2026-10-16", "2026-10-18")):  # fmt: skip
    read = []
    report = RunReport(date=day, manager="m")
    apply = apply or _Apply()
    _set_ahead(
        db, manager or _manager(), "L", None, day, WEEK, roster or _roster("a", "b"), None,
        report, moved_today=moved,
        read_rosters=lambda ds: read.append(ds) or {d: roster or _roster("a", "b") for d in ds},
        plan_for=_plan_for(changes_on), apply=apply,
    )  # fmt: skip
    return report, read, apply


def test_the_weeks_first_run_sets_each_later_day_that_needs_it(db):
    report, read, apply = _go(db)
    assert read == [["2026-10-16", "2026-10-17", "2026-10-18"]]  # only days after today
    assert apply.dates == ["2026-10-16", "2026-10-18"]  # 10-17 is already as planned
    step = report.steps[0]
    assert step.ok and "week's first run" in step.detail
    rows = db.execute("SELECT date FROM season_actions WHERE kind = 'lineup'").fetchall()
    assert [r[0] for r in rows] == ["2026-10-16", "2026-10-18"]  # each one recorded


def test_it_is_not_repeated_until_the_roster_changes(db):
    _go(db)
    report, read, apply = _go(db, day="2026-10-16")
    assert read == [] and apply.dates == [] and report.steps == []
    report, read, apply = _go(db, day="2026-10-16", roster=_roster("a", "c"))
    assert apply.dates == ["2026-10-18"] and "roster changed" in report.steps[0].detail


def test_an_ir_move_made_today_sets_it_again(db):
    _go(db)
    _, _, apply = _go(db, day="2026-10-16", moved=True)
    assert apply.dates == ["2026-10-18"]


def test_a_day_that_was_not_set_is_tried_again_next_run(db):
    report, _, _ = _go(db, apply=_Apply(ok=False))
    assert not report.steps[0].ok
    _, _, apply = _go(db)
    assert apply.dates == ["2026-10-16", "2026-10-18"]


def test_nothing_is_set_without_standing_authority_or_with_it_turned_off(db):
    for manager in (_manager(enabled=False), _manager(set_ahead=False)):
        report, read, apply = _go(db, manager=manager)
        assert read == [] and apply.dates == [] and report.steps == []


def test_the_weeks_last_day_has_nothing_ahead(db):
    report, read, _ = _go(db, day="2026-10-18")
    assert read == [] and report.steps == []
