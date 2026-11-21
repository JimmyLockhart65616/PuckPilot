"""One goalie count for the live plan, the add gate and the calibration."""

import pytest

from puckpilot.season.goalie_count import counted_starts, slot_chances


def _old_top(goalies, slots):
    """The rule as the three copies wrote it before they were merged."""
    gs = sorted(((p * v, p, pid) for v, p, pid, _ in goalies), reverse=True)[:slots]
    return {pid: p for _, p, pid in gs}


NIGHTS = [
    [(3.0, 0.6, 1, "AAA"), (2.0, 0.5, 2, "BBB")],
    [(3.0, 0.6, 1, "AAA"), (2.0, 0.5, 2, "BBB"), (2.5, 0.7, 3, "CCC")],
    [(3.0, 0.6, 1, "AAA"), (2.0, 0.5, 2, "BBB"), (2.5, 0.7, 3, "CCC"), (1.0, 0.9, 4, "DDD")],
    [(3.0, 0.6, 1, "AAA"), (2.0, 0.4, 2, "AAA"), (2.5, 0.7, 3, "CCC")],  # club-mates
    [(3.0, 0.0, 1, "AAA"), (2.0, 0.5, 2, "BBB"), (2.5, 0.7, 3, "CCC")],  # p = 0
    [(-1.0, 0.5, 1, "AAA"), (0.0, 0.0, 2, "BBB"), (2.5, 0.7, 3, "CCC")],  # a negative value
    [(2.0, 0.5, 7, "AAA"), (2.0, 0.5, 9, "BBB"), (2.0, 0.5, 8, "CCC")],  # ties
]


@pytest.mark.parametrize("night", NIGHTS)
@pytest.mark.parametrize("slots", [0, 1, 2, 3])
def test_top_is_the_rule_the_three_copies_wrote(night, slots):
    got = counted_starts(night, slots, "top")
    want = _old_top(night, slots)
    assert got == want
    assert list(got) == list(want)  # and in the same order: goalie games are summed in it


def test_min_is_slot_chances():
    night = NIGHTS[2]
    assert counted_starts(night, 2, "min") == slot_chances(night, 2)


def test_an_unknown_count_is_refused():
    with pytest.raises(ValueError):
        counted_starts(NIGHTS[0], 2, "all")
