"""The adds gate's plumbing: as-of values, and banked totals in Yahoo's labels."""

from __future__ import annotations

from puckpilot.season.add_gate import AsOfValues, as_labels
from puckpilot.season.week import banked_components


class _Seen:
    def __init__(self):
        self.asked: list[str] = []

    def per_game(self, pid, date):
        self.asked.append(date)
        return 1.0

    def per_game_tilted(self, pid, date, weights):
        self.asked.append(date)
        return 1.0

    def knows(self, pid):
        return True

    def projected(self, pid):
        return 1.0


def test_a_replayed_morning_cannot_see_later_in_the_week():
    """Asked about Saturday on Wednesday, the unclamped model blends in
    Thursday's and Friday's games."""
    seen = _Seen()
    v = AsOfValues(seen, "2026-10-07")
    v.per_game(1, "2026-10-10")
    v.per_game_tilted(1, "2026-10-09", {})
    v.per_game(1, "2026-10-05")
    assert seen.asked == ["2026-10-07", "2026-10-07", "2026-10-05"]


def test_banked_components_round_trip_through_yahoos_labels():
    comp = {
        "goals": 4.0,
        "assists": 6.0,
        "wins": 2.0,
        "saves": 180.0,
        "shots_against": 197.0,
        "goals_against": 17.0,
        "toi_hours": 5.0,
    }
    labels = as_labels(comp)
    assert labels["G"] == 4.0 and labels["SV"] == 180.0 and labels["SA"] == 197.0
    assert labels["GAA"] == 17.0 / 5.0
    back = banked_components(labels)
    for k in ("goals", "assists", "wins", "saves", "shots_against", "goals_against"):
        assert back[k] == comp[k]
    assert back["toi_hours"] == 5.0
