"""Market-implied value for players we cannot project at all.

`fit_market_curve` and `build_market_frame` are the only pieces of this feature
that touch a number (everyone else just renders `source`), so they get the
most scrutiny: a bad fit or a wrongly-skipped player is a silent mis-valuation,
exactly the failure mode `CLAUDE.md` calls out as this tool's worst.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from puckpilot.draft.engine import Universe
from puckpilot.draft.market import (
    MIN_CURVE_ROWS,
    build_market_frame,
    consensus_rank,
    fit_market_curve,
    mock_consensus,
    primary_position,
)

# ---- position parsing -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("LW,Util", "L"),
        ("RW,Util", "R"),
        ("C,LW,RW,Util", "C"),
        ("D,Util,IR+", "D"),
        ("G", "G"),
        ("G,IR+", "G"),
        ("Util,IR+", None),
        ("", None),
        (None, None),
    ],
)
def test_primary_position(raw, expected):
    assert primary_position(raw) == expected


# ---- mock consensus ---------------------------------------------------------


def _write_mock(path, n_teams, picks):
    path.write_text(
        json.dumps({"n_teams": n_teams, "rounds": 16, "completed": True, "picks": picks}),
        encoding="utf-8",
    )


def test_mock_consensus_normalizes_to_a_12_team_board(tmp_path):
    _write_mock(
        tmp_path / "a.json",
        n_teams=14,
        picks=[{"pick": 70, "yahoo_id": "111", "seat": 1, "position": "C"}],
    )
    out = mock_consensus(str(tmp_path / "*.json"))
    assert out["111"] == pytest.approx([60.0])  # 70 * 12 / 14


def test_mock_consensus_pools_across_files(tmp_path):
    _write_mock(tmp_path / "a.json", 12, [{"pick": 50, "yahoo_id": "111"}])
    _write_mock(tmp_path / "b.json", 12, [{"pick": 55, "yahoo_id": "111"}])
    out = mock_consensus(str(tmp_path / "*.json"))
    assert sorted(out["111"]) == [50.0, 55.0]


def test_mock_consensus_skips_an_unparsable_file(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    _write_mock(tmp_path / "ok.json", 12, [{"pick": 1, "yahoo_id": "111"}])
    out = mock_consensus(str(tmp_path / "*.json"))
    assert out == {"111": [1.0]}


def test_mock_consensus_ignores_picks_missing_an_id_or_number(tmp_path):
    _write_mock(
        tmp_path / "a.json",
        12,
        [{"pick": 1}, {"yahoo_id": "111"}, {"pick": 2, "yahoo_id": "222"}],
    )
    out = mock_consensus(str(tmp_path / "*.json"))
    assert out == {"222": [2.0]}


# ---- consensus_rank ----------------------------------------------------------


def test_no_mock_evidence_is_pure_yahoo_adp():
    assert consensus_rank(92.0, []) == 92.0


def test_full_evidence_is_pure_mock_mean():
    picks = [70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0]  # 8 observations
    assert consensus_rank(999.0, picks) == pytest.approx(sum(picks) / len(picks))


def test_partial_evidence_blends_proportionally():
    # 4 of the 8-observation full-weight threshold -> exactly half-and-half
    picks = [50.0, 50.0, 50.0, 50.0]
    assert consensus_rank(100.0, picks) == pytest.approx(75.0)


def test_more_than_full_weight_does_not_overshoot():
    picks = [10.0] * 20
    assert consensus_rank(999.0, picks) == pytest.approx(10.0)


# ---- fit_market_curve --------------------------------------------------------


def _synthetic_population(n_per_pos=8, noise=0.0, rng=None):
    """(vorp, adp_rank, position, has_market) with a KNOWN log-linear shape:
    vorp = intercept[pos] - 4 * log(adp_rank), so a fit can be checked exactly."""
    rng = rng or np.random.default_rng(0)
    intercepts = {"C": 15.0, "D": 20.0, "L": 18.0, "R": 18.0}
    vorp, adp, pos = [], [], []
    rank = 1
    for p, b0 in intercepts.items():
        for _ in range(n_per_pos):
            rank += rng.integers(1, 4)
            v = b0 - 4.0 * np.log(rank) + (rng.normal(0, noise) if noise else 0.0)
            vorp.append(v)
            adp.append(float(rank))
            pos.append(p)
    return (
        np.array(vorp),
        np.array(adp),
        np.array(pos),
        np.ones(len(vorp), dtype=bool),
    )


def test_curve_recovers_a_known_shape():
    vorp, adp, pos, has_market = _synthetic_population()
    curve = fit_market_curve(vorp, adp, pos, has_market)
    assert curve is not None
    assert curve.slope == pytest.approx(-4.0, abs=0.05)
    assert curve.intercept["D"] - curve.intercept["C"] == pytest.approx(5.0, abs=0.1)
    assert curve.r_squared > 0.99


def test_too_few_rows_refuses_to_fit():
    vorp, adp, pos, has_market = _synthetic_population(n_per_pos=2)
    assert len(vorp) < MIN_CURVE_ROWS
    assert fit_market_curve(vorp, adp, pos, has_market) is None


def test_a_single_usable_position_refuses_to_fit():
    """One intercept alone is not a position effect - it is just the slope
    fit's constant term wearing a position's name."""
    vorp, adp, pos, has_market = _synthetic_population(n_per_pos=25)
    only_d = pos == "D"
    assert fit_market_curve(vorp[only_d], adp[only_d], pos[only_d], has_market[only_d]) is None


def test_goalies_are_excluded_by_default():
    vorp, adp, pos, has_market = _synthetic_population()
    # Corrupt goalie-shaped rows into the pool with a shape the real curve
    # would badly mis-fit if it looked at them at all.
    vorp = np.concatenate([vorp, np.full(10, 50.0)])
    adp = np.concatenate([adp, np.arange(1, 11, dtype=float)])
    pos = np.concatenate([pos, np.full(10, "G")])
    has_market = np.concatenate([has_market, np.ones(10, dtype=bool)])
    curve = fit_market_curve(vorp, adp, pos, has_market)
    assert curve is not None
    assert "G" not in curve.intercept


def test_only_market_priced_rows_are_used():
    vorp, adp, pos, has_market = _synthetic_population()
    # Throw in junk rows marked NOT market-priced - if they leaked in, the
    # clean synthetic shape would no longer fit almost perfectly.
    junk = np.full(30, 500.0)
    vorp2 = np.concatenate([vorp, junk])
    adp2 = np.concatenate([adp, junk])
    pos2 = np.concatenate([pos, np.full(30, "C")])
    has_market2 = np.concatenate([has_market, np.zeros(30, dtype=bool)])
    curve = fit_market_curve(vorp2, adp2, pos2, has_market2)
    assert curve is not None
    assert curve.r_squared > 0.99


# ---- build_market_frame, end to end -----------------------------------------

LEAGUE_KEY = "477.l.1"


def _seed_universe(n_per_pos=8):
    """A `Universe` shaped like a real live board: real projected players,
    Yahoo ADP already applied, `has_market` already set - the state
    `build_live_board` hands to `build_market_frame`."""
    rng = np.random.default_rng(1)
    intercepts = {"C": 15.0, "D": 20.0, "L": 18.0, "R": 18.0}
    rows, pid, rank = {}, 1, 0
    for p, b0 in intercepts.items():
        for _ in range(n_per_pos):
            rank += int(rng.integers(1, 4))
            v = b0 - 4.0 * np.log(rank)
            rows[pid] = {
                "name": f"Real {p}{pid}",
                "position": p,
                "team": "AAA",
                "vorp": v,
                "z_total": v + 3.0,  # a fixed, checkable replacement-level offset
                "adp_rank": float(rank),
            }
            pid += 1
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    u = Universe(df.sort_values("vorp", ascending=False))
    u.has_market = np.ones(len(u), dtype=bool)
    return u


def _seed_map(conn, rows):
    conn.executemany(
        "INSERT INTO yahoo_player_map"
        " (player_key, league_key, full_name, team_abbrev, positions, nhl_player_id, adp_rank)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(key, LEAGUE_KEY, *rest) for key, *rest in rows],
    )
    conn.commit()


def test_an_unprojectable_market_priced_player_gets_a_row(db):
    u = _seed_universe()
    _seed_map(
        db,
        [
            ("477.p.900", "Rookie Winger", "BOS", "LW,Util", None, 90),
        ],
    )
    frame = build_market_frame(db, LEAGUE_KEY, u)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["position"] == "L"
    assert row["source"] == "market"
    assert row["train_gp"] == 0.0
    assert np.isfinite(row["vorp"])


def test_a_player_already_on_the_board_is_not_duplicated(db):
    u = _seed_universe()
    real_id = int(u.ids[0])
    _seed_map(db, [("477.p.1", "Real Player", "AAA", "C,Util", real_id, 5)])
    frame = build_market_frame(db, LEAGUE_KEY, u)
    assert frame.empty


def test_a_goalie_with_no_projection_is_skipped(db):
    """The unprojectable population beyond skaters is backup goalies nobody
    starts - excluded rather than priced off a curve fit without them."""
    u = _seed_universe()
    _seed_map(db, [("477.p.901", "Deep Goalie", "BOS", "G", None, 390)])
    frame = build_market_frame(db, LEAGUE_KEY, u)
    assert frame.empty


def test_a_player_with_no_resolvable_position_is_skipped(db):
    u = _seed_universe()
    _seed_map(db, [("477.p.902", "No Position", "BOS", "Util,IR+", None, 200)])
    frame = build_market_frame(db, LEAGUE_KEY, u)
    assert frame.empty


def test_mock_consensus_shifts_the_implied_rank(tmp_path, db):
    """The whole point of blending: a player drafted consistently earlier in
    real mocks than his Yahoo ADP must land a better implied VORP than ADP
    alone would give him — not the same number twice."""
    u = _seed_universe()
    _seed_map(db, [("477.p.900", "Rookie Winger", "BOS", "LW,Util", None, 150)])
    for i in range(8):
        _write_mock(tmp_path / f"mock{i}.json", 12, [{"pick": 60, "yahoo_id": "900"}])

    plain = build_market_frame(db, LEAGUE_KEY, u, mock_glob=str(tmp_path / "none-*.json"))
    boosted = build_market_frame(db, LEAGUE_KEY, u, mock_glob=str(tmp_path / "*.json"))

    assert boosted.iloc[0]["adp_rank"] < plain.iloc[0]["adp_rank"]
    assert boosted.iloc[0]["vorp"] > plain.iloc[0]["vorp"]


def test_a_thin_market_refuses_to_fit_and_returns_nothing(db):
    """Fewer than `MIN_CURVE_ROWS` market-priced rows must not produce a curve
    fit to noise - see `fit_market_curve`."""
    from puckpilot.draft.engine import Universe as U

    rows = {
        1: {
            "name": "A",
            "position": "C",
            "team": "AAA",
            "vorp": 5.0,
            "z_total": 5.0,
            "adp_rank": 1.0,
        },
        2: {
            "name": "B",
            "position": "D",
            "team": "AAA",
            "vorp": 3.0,
            "z_total": 3.0,
            "adp_rank": 2.0,
        },
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "player_id"
    u = U(df)
    u.has_market = np.ones(len(u), dtype=bool)
    _seed_map(db, [("477.p.900", "Rookie", "BOS", "LW,Util", None, 90)])
    assert build_market_frame(db, LEAGUE_KEY, u).empty


def test_z_total_is_backfilled_from_the_real_replacement_offset(db):
    u = _seed_universe()  # z_total = vorp + 3.0 for every real row, by construction
    _seed_map(db, [("477.p.900", "Rookie Winger", "BOS", "LW,Util", None, 90)])
    frame = build_market_frame(db, LEAGUE_KEY, u)
    row = frame.iloc[0]
    assert row["z_total"] == pytest.approx(row["vorp"] + 3.0, abs=1e-6)
