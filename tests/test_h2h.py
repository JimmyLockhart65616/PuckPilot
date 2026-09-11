import numpy as np
import pytest

from puckpilot.draft.h2h import round_robin_schedule, run_h2h_season, score_matchup
from puckpilot.draft.replay import G_GA, G_HOURS, G_SA, G_WIDTH, G_WINS
from puckpilot.engine.categories import Category

GOALS = Category("goals", "G", "skater")
WINS = Category("wins", "W", "goalie")
GAA = Category("gaa", "GAA", "goalie", higher_is_better=False, rate=True)


def test_configured_keeper_names_all_resolve(db):
    """A keeper name that fails to resolve would leave an elite player wrongly
    available on the draft board, so this must never silently pass.

    Runs against whatever LEAGUE_FILE points at, and skips when there is none.
    League configs are private - they carry a live league id and named keeper
    rosters - so a clone of this public repo has no such file, and naming one
    here would put the league's identity back into tracked source. The mechanism
    is covered against a fixture in tests/test_league_config.py; this is the
    check against a real keeper list.
    """
    from puckpilot.config import Settings
    from puckpilot.data import store
    from puckpilot.keepers import keeper_seats, resolve_keeper_ids
    from puckpilot.league import load_league

    league_file = Settings().resolved_league_path
    if not league_file.is_file():
        pytest.skip("no league config present (LEAGUE_FILE unset or private file absent)")

    league = load_league(league_file)
    names = league.keepers_for_season("20262027")
    assert len(names) > 0
    for i, n in enumerate(names):
        store.upsert_player(db, 9000 + i, n, "C", "AAA")
    # accents and spelling variants must fold to the same key
    store.upsert_player(db, 8000, "Tim Stützle", "C", "AAA")

    resolved, unmatched = resolve_keeper_ids(db, (*names, "Tim Stutzle"))
    assert unmatched == []
    assert len(resolved) == len(names) + 1

    seats = keeper_seats(list(resolved.values()), 12, np.random.default_rng(0))
    assert sum(len(v) for v in seats.values()) == len(resolved)
    assert max(len(v) for v in seats.values()) - min(len(v) for v in seats.values()) <= 1


def test_round_robin_every_team_plays_once_per_week():
    for week in round_robin_schedule(12, 19):
        assert len(week) == 6
        played = [t for pair in week for t in pair]
        assert sorted(played) == list(range(12))


def test_round_robin_rotates_opponents():
    weeks = round_robin_schedule(12, 11)
    opponents = set()
    for week in weeks:
        for a, b in week:
            if a == 0 or b == 0:
                opponents.add(b if a == 0 else a)
    assert opponents == set(range(1, 12))  # a full cycle faces everyone once


def test_score_matchup_counts_wins_losses_ties():
    a = np.array([5.0, 2.0, 3.0])
    b = np.array([1.0, 9.0, 3.0])
    assert score_matchup(a, b) == (1, 1, 1)


def _weekly(goals_per_week, wins_per_week, n_weeks, ga=1.0):
    """(T, weeks, ...) arrays where team t scores goals_per_week[t] each week."""
    n = len(goals_per_week)
    sk = np.zeros((n, n_weeks, 1))
    g = np.zeros((n, n_weeks, G_WIDTH))
    for t in range(n):
        sk[t, :, 0] = goals_per_week[t]
        g[t, :, G_WINS] = wins_per_week[t]
        g[t, :, G_GA] = ga
        g[t, :, G_SA] = 30.0
        g[t, :, G_HOURS] = 1.0
    return sk, g


def test_dominant_team_takes_top_seed_and_title():
    sk, g = _weekly([1, 2, 3, 4], [1, 2, 3, 4], n_weeks=5)
    res = run_h2h_season(
        sk,
        g,
        (GOALS,),
        (WINS,),
        ["goals"],
        regular_weeks=4,
        playoff_teams=2,
        playoff_weeks=1,
    )
    best = 3  # highest in every category every week
    assert res.seeds[best] == 1
    assert res.champion == best
    assert res.finish[best] == 1
    # undefeated across 4 weeks of a 4-team round robin
    assert res.records[best][0] == 4
    assert res.records[best][1] == 0


def test_records_are_symmetric_across_the_league():
    sk, g = _weekly([1, 2, 3, 4], [4, 3, 2, 1], n_weeks=6)
    res = run_h2h_season(
        sk,
        g,
        (GOALS,),
        (WINS,),
        ["goals"],
        regular_weeks=5,
        playoff_teams=2,
        playoff_weeks=1,
    )
    # every matchup produces one win and one loss (or two ties)
    assert res.records[:, 0].sum() == res.records[:, 1].sum()
    assert res.records.sum() == 4 * 5  # 4 teams x 5 weeks of results
    assert sorted(res.finish) == [1, 2, 3, 4]


def test_lower_is_better_category_is_inverted():
    sk = np.zeros((2, 3, 1))
    g = np.zeros((2, 3, G_WIDTH))
    g[:, :, G_HOURS] = 1.0
    g[0, :, G_GA] = 1.0  # team 0 allows fewer goals -> should win GAA
    g[1, :, G_GA] = 5.0
    res = run_h2h_season(
        sk,
        g,
        (GOALS,),
        (GAA,),
        ["goals"],
        regular_weeks=2,
        playoff_teams=2,
        playoff_weeks=1,
    )
    assert res.seeds[0] == 1
    assert res.cat_records[0][0] > res.cat_records[1][0]
