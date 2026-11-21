"""The replay league beside the live one."""

from __future__ import annotations

import json
from types import SimpleNamespace

from puckpilot.season import realism


def test_the_live_league_profile_reads_its_own_tables(db):
    week = SimpleNamespace(number=1, start="2026-10-06", end="2026-10-12")
    rt = SimpleNamespace(
        league_key="999.l.1", weeks=[week], stat_labels=lambda: {25: "SV", 19: "W"}
    )
    for team, sv, w in (("a", 120, 3), ("b", 80, 1)):
        db.execute(
            "INSERT INTO league_week_stats (league_key, week, team_key, name, opponent_key,"
            " status, cats_won, completed_games, stats_json, fetched_at)"
            " VALUES ('999.l.1', 1, ?, ?, '', 'postevent', 0, 0, ?, '')",
            (team, team, json.dumps({"25": sv, "19": w})),
        )
    for key, elig in (("g1", "G"), ("g2", "G"), ("s1", "C,LW")):
        db.execute(
            "INSERT INTO yahoo_roster_snapshots (manager, league_key, team_key, date, player_key,"
            " name, eligible) VALUES ('m', '999.l.1', 'a', '2026-10-06', ?, ?, ?)",
            (key, key, elig),
        )
    db.execute(
        "INSERT INTO league_transactions (league_key, txn_key, at, type, team_key, action,"
        " player_key, positions) VALUES ('999.l.1', 't1', '2026-10-08T12:00:00Z', 'add', 'b',"
        " 'add', 'g9', 'G')"
    )
    got = realism.live_profile(db, rt)
    assert got["goalies held"] == [2.0]  # b had no saved roster that day: not counted
    assert got["saves"] == [120.0, 80.0] and got["wins"] == [3.0, 1.0]
    assert got["goalie adds"] == [0.0, 1.0] and got["goalie appearances"] == []
    text = realism.text({"saves": [100.0, 90.0]}, got, "20252026")
    saves = next(ln for ln in text.splitlines() if ln.strip().startswith("saves"))
    assert "replayed 20252026 league" in text
    assert "95.00" in saves and "100.00" in saves  # replay mean of 100, 90; live of 120, 80
