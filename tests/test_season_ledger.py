"""The move ledger: an executed move's nights in the lineup, against the player let go."""

from __future__ import annotations

import json

from puckpilot.engine.categories import CATALOG
from puckpilot.season import ledger
from puckpilot.season.fetch import save_roster
from puckpilot.season.roster import RosterPlayer, TeamRoster
from tests.conftest import add_goalie_game, add_skater_game

SEASON = "20262027"


def _rp(key, pid, slot, pos="D"):
    return RosterPlayer(player_key=key, yahoo_id=key, name=key.title(), team="OTT",
                        primary_position=pos, yahoo_eligible=frozenset({pos}),
                        selected_slot=slot, nhl_player_id=pid)  # fmt: skip


def test_a_move_is_judged_on_the_nights_it_was_for(db):
    # Executed 10-12 at 18:08 local-ish (22:08Z): it first counts on 10-13.
    db.execute(
        "INSERT INTO waiver_proposals (add_pid, drop_pid, reason_json, status, manager, "
        "league_key, team_key, kind, add_player_key, drop_player_key, executed_at) VALUES "
        "(1, 2, ?, 'executed', 'm', 'L', 'T', 'add_drop', 'skater', 'goalie', "
        "'2026-10-12T22:08:11+00:00')",
        (json.dumps({"add_name": "Skater", "drop_name": "Goalie"}),),
    )
    db.commit()
    for day, slot in (("2026-10-13", "D"), ("2026-10-14", "BN"), ("2026-10-15", "D")):
        save_roster(db, "m", TeamRoster(league_key="L", team_key="T", date=day,
                                        players=(_rp("skater", 1, slot),)))  # fmt: skip
    save_roster(db, "m", TeamRoster(league_key="L", team_key="T", date="2026-10-16",
                                    players=(_rp("other", 3, "D"),)))  # fmt: skip
    add_skater_game(db, 1, SEASON, 10, date="2026-10-13", goals=1, shots=3, blocks=2)
    add_skater_game(db, 1, SEASON, 11, date="2026-10-14", goals=5)  # benched: not counted
    add_skater_game(db, 1, SEASON, 12, date="2026-10-15", shots=2)
    add_skater_game(db, 1, SEASON, 13, date="2026-10-16", goals=9)  # gone: not counted
    add_goalie_game(db, 2, SEASON, 20, date="2026-10-13", shots_against=30, goals_against=2,
                    decision="W")  # fmt: skip
    db.commit()
    (line,) = ledger.build(db, "m", "L", "T", SEASON, today="2026-10-20")
    assert line.made_on in ("2026-10-12", "2026-10-13")
    assert line.nights == 2
    assert line.add_totals["goals"] == 1 and line.add_totals["sog"] == 5
    assert line.drop_nights == 1 and line.drop_totals["saves"] == 28
    text = "\n".join(ledger.describe(line, [CATALOG[k] for k in ("G", "SOG", "W", "SV", "SV%")]))
    assert "2 night(s) in the lineup" in text and "SV 28" in text and "SV% 0.933" in text
