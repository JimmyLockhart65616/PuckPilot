"""Fitting survival_spread to real draft behaviour.

`survival_spread = 6.0` is currently a guess fitted against a *simulated* ADP
bot field, and STATUS.md credits `survival_discount` with the engine's whole
measured edge — so before this number is allowed to move, the estimator has to
prove it can recover a spread it was never told.

The recovery test below matters more than it looks. Two earlier versions of it
reported the estimator as biased by 25-30%; both times the fault was in the
synthetic *generator*, not the fitter. The generator here derives the pick
position in closed form and does not reassign colliding picks, because shifting
a collided player to a later pick inflates observed survival and pushes the fit
wide.
"""

from __future__ import annotations

import math
import random

from puckpilot.draft.calibrate import (
    Observation,
    _log_loss,
    calibrate,
    observations_from,
)
from puckpilot.draft.farm import MockResult


def synthetic_draft(true_spread: float, n: int = 400, seed: int = 4) -> MockResult:
    """A draft where P(survive to pick P) really is logistic((rank-P)/spread).

    Closed form: a player drawn with uniform u is taken at
    p* = rank - spread * logit(u), which gives exactly that survival curve.
    """
    rng = random.Random(seed)
    picks, adp = [], []
    for pid in range(1, n):
        rank = float(pid)
        u = min(max(rng.random(), 1e-6), 1 - 1e-6)
        gone = max(1, int(round(rank - true_spread * math.log(u / (1 - u)))))
        picks.append({"pick": gone, "yahoo_id": str(pid), "seat": 1, "position": "C"})
        adp.append({"yahoo_id": str(pid), "adp_rank": rank})
    return MockResult(started="t", picks=picks, adp_observations=adp, n_teams=12)


# ---- the estimator can find a spread it was not told ----------------------


def test_recovers_a_known_spread():
    for true_spread in (3.0, 4.0, 6.0, 10.0):
        best = calibrate([synthetic_draft(true_spread)]).best
        assert best == true_spread, f"spread {true_spread} recovered as {best}"


def test_recovery_is_stable_across_seeds():
    for seed in (1, 7, 23):
        assert calibrate([synthetic_draft(6.0, seed=seed)]).best == 6.0


def test_log_loss_is_minimised_at_the_truth():
    """The grid is a means; the loss surface is the actual claim."""
    obs = observations_from(synthetic_draft(6.0))
    losses = {s: _log_loss(obs, s) for s in (3.0, 6.0, 13.0)}
    assert losses[6.0] < losses[3.0] and losses[6.0] < losses[13.0]


# ---- observation construction --------------------------------------------


def _draft_of(pairs, last_pick=200):
    """A draft containing `pairs` of (yahoo_id, adp, pick), padded so the draft
    genuinely runs to `last_pick`."""
    picks = [{"pick": pk, "yahoo_id": pid, "seat": 1, "position": "C"} for pid, _, pk in pairs]
    picks.append({"pick": last_pick, "yahoo_id": "filler", "seat": 1, "position": "C"})
    adp = [{"yahoo_id": pid, "adp_rank": a} for pid, a, _ in pairs]
    return MockResult(started="t", picks=picks, adp_observations=adp, n_teams=12)


def test_observations_are_centred_on_adp_not_on_the_outcome():
    """Anchoring the window on when a player went, and capping only the forward
    side, over-samples 'he lasted' relative to reaches and biases the fit wide —
    measured at +25-30% before this was corrected."""
    obs = observations_from(_draft_of([("1", 100.0, 5)]), max_lookahead=30)
    mine = [o for o in obs if o.adp_rank == 100.0]
    targets = [o.target_pick for o in mine]
    # centred on ADP 100, not on the pick-5 outcome
    assert min(targets) == 70 and max(targets) == 130
    # a big reach shows as "gone" across nearly the whole window
    assert sum(1 for o in mine if o.survived) <= 1


def test_survived_is_true_only_up_to_the_pick_he_went():
    obs = {
        o.target_pick: o.survived
        for o in observations_from(_draft_of([("1", 50.0, 50)]), max_lookahead=5)
        if o.adp_rank == 50.0
    }
    assert obs[48] is True and obs[50] is True and obs[51] is False


def test_the_window_stops_at_the_end_of_the_draft():
    """We cannot observe a pick that never happened. Late-ADP players therefore
    contribute a one-sided window — a real limitation, not an oversight."""
    obs = observations_from(_draft_of([("1", 95.0, 90)], last_pick=100), max_lookahead=30)
    assert max(o.target_pick for o in obs) == 100


def test_a_player_whose_adp_is_past_the_draft_yields_nothing():
    obs = observations_from(_draft_of([("1", 400.0, 12)], last_pick=100), max_lookahead=30)
    assert [o for o in obs if o.adp_rank == 400.0] == []


def test_falls_back_to_draft_order_when_yahoo_publishes_no_adp():
    """Without published ADP, when a player went is the only statement the
    draft makes about where the market valued him."""
    result = MockResult(
        started="t",
        picks=[
            {"pick": 1, "yahoo_id": "a", "seat": 1, "position": "C"},
            {"pick": 2, "yahoo_id": "b", "seat": 2, "position": "C"},
        ],
        adp_observations=[],
        n_teams=12,
    )
    assert observations_from(result), "draft order must stand in for ADP"


def test_a_draft_with_no_picks_yields_nothing():
    assert observations_from(MockResult(started="t")) == []


# ---- reporting -------------------------------------------------------------


def test_report_says_so_when_there_is_nothing_to_fit():
    report = calibrate([])
    assert report.n_observations == 0
    assert "Harvest mock drafts first" in report.text


def test_report_keeps_the_incumbent_when_the_data_agrees():
    report = calibrate([synthetic_draft(6.0)], incumbent=6.0)
    assert report.best == 6.0
    assert "Keep survival_spread = 6.0" in report.text


def test_report_warns_that_one_draft_is_not_enough_to_move_a_knob():
    report = calibrate([synthetic_draft(16.0)], incumbent=6.0)
    assert report.best != 6.0
    assert "Only 1 draft(s) harvested" in report.text


def test_report_calls_a_tiny_margin_agreement_not_a_mandate():
    """A hair's difference in log-loss is not evidence to retune a constant the
    draft sim arbitrated."""
    report = calibrate([synthetic_draft(6.0)], incumbent=6.0)
    report.best = 8.0
    report.losses[8.0] = report.losses[6.0] - 0.001
    assert "small" in report.text


def test_log_loss_survives_extreme_offsets():
    """A player 500 picks from the boundary must not overflow the exponent."""
    obs = [Observation(1.0, 900, False), Observation(900.0, 1, True)]
    assert math.isfinite(_log_loss(obs, 6.0))


# ---- harvesting ------------------------------------------------------------


def test_an_abandoned_lobby_is_not_written_to_disk(tmp_path):
    """A mock that never started yields zero picks. Saving it would inflate the
    draft count the report leans on when it says how much evidence there is."""
    from puckpilot.draft.farm import load_all, save

    assert save(MockResult(started="t"), tmp_path) is None
    assert load_all(tmp_path) == []


def test_a_real_harvest_round_trips(tmp_path):
    from puckpilot.draft.farm import load_all, save

    result = synthetic_draft(6.0, n=20)
    assert save(result, tmp_path) is not None
    back = load_all(tmp_path)
    assert len(back) == 1 and len(back[0].picks) == len(result.picks)
