"""Standing authority for lineups; approval as a precondition for transactions.

The asymmetry is the feature, so these mostly pin that it cannot be configured
away.
"""

from __future__ import annotations

import pytest

from puckpilot.season.authority import (
    Authority,
    AuthorityError,
    LineupAuthority,
    TransactionAuthority,
)


def test_lineup_authority_is_off_until_granted():
    assert LineupAuthority().enabled is False
    assert "RECOMMEND ONLY" in "\n".join(LineupAuthority().describe())


def test_granting_it_prints_the_criteria_it_was_granted_under():
    text = "\n".join(LineupAuthority(enabled=True, min_gain=0.2).describe())
    assert "AUTONOMOUS" in text
    assert "0.20" in text


def test_there_is_no_cap_on_changes_per_day():
    """Lineup moves are free and unlimited until each player's game starts, so
    a per-day cap modelled a constraint that does not exist."""
    assert "max_swaps_per_day" not in LineupAuthority.__dataclass_fields__
    assert "unlimited until" in "\n".join(LineupAuthority(enabled=True).describe())


def test_a_config_still_carrying_the_old_cap_is_told_why_it_went():
    with pytest.raises(AuthorityError, match="has been removed"):
        Authority.from_config({"lineup": {"max_swaps_per_day": 4}})


def test_never_bench_is_listed_when_set():
    text = "\n".join(LineupAuthority(enabled=True, never_bench=("A Player",)).describe())
    assert "never benched: A Player" in text


# -- transactions cannot be made autonomous ---------------------------------


def test_transactions_always_require_approval():
    assert TransactionAuthority().requires_approval is True


def test_requires_approval_is_not_a_field_that_can_be_set():
    """A property, so there is nothing to pass to the constructor."""
    assert "requires_approval" not in TransactionAuthority.__dataclass_fields__
    with pytest.raises(TypeError):
        TransactionAuthority(requires_approval=False)


def test_an_enabled_flag_on_transactions_is_an_error_not_a_setting():
    with pytest.raises(AuthorityError, match="not a setting"):
        Authority.from_config({"transactions": {"enabled": True}})


def test_the_description_always_says_approval_required():
    assert "APPROVAL REQUIRED" in Authority().describe()
    assert "APPROVAL REQUIRED" in Authority(lineup=LineupAuthority(enabled=True)).describe()


# -- config validation ------------------------------------------------------


def test_a_misspelled_bound_is_refused_rather_than_silently_defaulted():
    """Acting under criteria nobody agreed to is the failure mode here."""
    with pytest.raises(AuthorityError, match="unknown setting"):
        Authority.from_config({"lineup": {"min_gainn": 0.5}})


def test_questionable_mode_is_validated():
    with pytest.raises(AuthorityError, match="start_questionable"):
        LineupAuthority(start_questionable="probably")


def test_goalie_probability_must_be_a_probability():
    with pytest.raises(AuthorityError, match="between 0 and 1"):
        LineupAuthority(min_goalie_p_start=1.5)


def test_never_bench_becomes_a_tuple_from_a_toml_list():
    a = Authority.from_config({"lineup": {"never_bench": ["One", "Two"]}})
    assert a.lineup.never_bench == ("One", "Two")


def test_an_empty_config_is_the_safe_default():
    a = Authority.from_config({})
    assert a.lineup.enabled is False
    assert a.transactions.requires_approval is True


def test_the_protocol_does_not_steer_the_lineup_unless_asked():
    """Measured inert: scored on categories won per week it was positive in 3
    of 6 runs across two seasons and three seeds, every effect inside 0.08 of
    ~5.8. Most weeks everyone with a game fits in a slot, so there is no
    lineup decision for a weight to change."""
    assert LineupAuthority().follow_protocol is False
    text = "\n".join(LineupAuthority(enabled=True).describe())
    assert "measured inert" in text


def test_opting_in_is_possible_and_says_so():
    text = "\n".join(LineupAuthority(enabled=True, follow_protocol=True).describe())
    assert "week protocol steers the lineup: yes" in text
