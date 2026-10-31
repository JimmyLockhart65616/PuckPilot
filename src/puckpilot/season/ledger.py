"""The move ledger: what each move PuckPilot made actually did.

A move is priced before it is made; this is the after. For every executed
move, the nights the player brought in filled a starting slot and played are
the nights the move was for, and on those same nights the player let go - who
kept playing for some other team, or for nobody - is the counterfactual. Both
are real games, so the ledger needs no model and has nothing to argue about:

    #15 Tyler Kleven for John Gibson, from 10-13: 8 nights in the lineup
        Tyler Kleven   G 1  A 0  SOG 12  HIT 13  BLK 20
        John Gibson    W 3  SV 106  (5 starts on those nights)

It is the tool's own track record - the evidence another manager would ask
for before letting it near a roster - and it is honest about what it is not:
the drop's games count only on nights the add was slotted, so a drop who
would have sat on our bench is credited as if he would have played.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, timedelta

from puckpilot.draft.replay import BOX_KEYS, LOG_KEYS
from puckpilot.season.settings import BENCH_SLOTS, IR_SLOTS

# Executed after this hour (local), a move first counts the next day: the
# evening's games had begun, or the drop had already played.
LATE_HOUR = 17


@dataclass
class LedgerLine:
    proposal_id: int
    kind: str
    made_on: str
    add_name: str
    drop_name: str
    nights: int = 0
    add_totals: dict[str, float] = field(default_factory=dict)
    drop_totals: dict[str, float] = field(default_factory=dict)
    drop_nights: int = 0

    def delta(self, key: str) -> float:
        return self.add_totals.get(key, 0.0) - self.drop_totals.get(key, 0.0)


def _line(stats: dict, box: dict) -> dict[str, float]:
    """One game's counting stats, by category key (goalie games by their own keys)."""
    if "shotsAgainst" in stats:
        sa = float(stats.get("shotsAgainst") or 0)
        ga = float(stats.get("goalsAgainst") or 0)
        return {
            "wins": 1.0 if stats.get("decision") == "W" else 0.0,
            "saves": sa - ga,
            "shots_against": sa,
            "goals_against": ga,
            "starts": float(stats.get("gamesStarted") or 0),
        }
    out = {k: float(stats.get(v) or 0) for k, v in LOG_KEYS.items()}
    out.update({k: float(box.get(v) or 0) for k, v in BOX_KEYS.items()})
    return out


def _games(conn, season: str, pids) -> dict[tuple[int, str], dict[str, float]]:
    pids = sorted({p for p in pids if p is not None})
    if not pids:
        return {}
    marks = ",".join("?" * len(pids))
    rows = conn.execute(
        f"SELECT g.player_id, g.game_date, g.stats_json, b.stats_json FROM nhl_game_logs g "
        f"LEFT JOIN nhl_boxscore_stats b ON b.player_id = g.player_id AND b.game_id = g.game_id "
        f"WHERE g.season = ? AND g.player_id IN ({marks})",
        (season, *pids),
    )
    return {(pid, day): _line(json.loads(s), json.loads(b) if b else {}) for pid, day, s, b in rows}


def _slots(conn, team_key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for day, key, slot in conn.execute(
        "SELECT date, player_key, selected_slot FROM yahoo_roster_snapshots WHERE team_key = ?",
        (team_key,),
    ):
        out.setdefault(day, {})[key] = slot or ""
    return out


def _first_day(executed_at: str) -> str:
    when = datetime.fromisoformat(executed_at.replace("Z", "+00:00")).astimezone()
    day = when.date()
    return (day + timedelta(days=1) if when.hour >= LATE_HOUR else day).isoformat()


def build(conn, manager: str, league_key: str, team_key: str, season: str, today: str):
    """One line per executed move, oldest first."""
    from puckpilot.season import proposals as proposals_mod

    made = [
        p
        for p in proposals_mod.listing(conn, manager, league_key, status=proposals_mod.EXECUTED,
                                       limit=500)
        if p.executed_at
    ]  # fmt: skip
    made.sort(key=lambda p: p.executed_at)
    games = _games(conn, season, [p.add_pid for p in made] + [p.drop_pid for p in made])
    slots = _slots(conn, team_key)
    lines = []
    for p in made:
        line = LedgerLine(p.id, p.kind, _first_day(p.executed_at), p.add_name,
                          p.drop_name if p.drop_player_key else "")  # fmt: skip
        day = _date.fromisoformat(line.made_on)
        end = _date.fromisoformat(today)
        while day <= end:
            d = day.isoformat()
            slot = slots.get(d, {}).get(p.add_player_key)
            if slot is None and d in slots:
                break  # he has left the roster: the move's run is over
            started = slot is not None and slot not in BENCH_SLOTS and slot not in IR_SLOTS
            mine = games.get((p.add_pid, d))
            if started and mine is not None:  # a goalie's relief night counts, as in Yahoo
                line.nights += 1
                for k, v in mine.items():
                    line.add_totals[k] = line.add_totals.get(k, 0.0) + v
                theirs = games.get((p.drop_pid, d)) if p.drop_pid else None
                if theirs is not None:
                    line.drop_nights += 1
                    for k, v in theirs.items():
                        line.drop_totals[k] = line.drop_totals.get(k, 0.0) + v
            day += timedelta(days=1)
        lines.append(line)
    return lines


def summary(line: LedgerLine, cats) -> dict:
    """One move for the page: what it was, and both sides' lines in the
    league's categories - the categories only, the zeros left out."""

    def short(totals: dict[str, float]) -> str:
        bits = []
        for c in cats:
            if c.key == "save_pct":
                sa = totals.get("shots_against", 0.0)
                if sa:
                    bits.append(f"{c.label} {totals.get('saves', 0.0) / sa:.3f}")
            elif totals.get(c.key):
                bits.append(f"{c.label} {totals[c.key]:g}")
        return " ".join(bits) or "nothing"

    head = f"#{line.proposal_id} {line.add_name}" + (
        f" for {line.drop_name}" if line.drop_name else " into an open spot"
    )
    out = {
        "head": head,
        "since": line.made_on,
        "nights": line.nights,
        "add": f"{line.add_name}: {short(line.add_totals)}",
        "drop": "",
    }
    if line.drop_name:
        out["drop"] = f"{line.drop_name} on those nights: {short(line.drop_totals)}"
    return out


def describe(line: LedgerLine, cats) -> list[str]:
    """The ledger line in the league's own categories."""

    def fmt(totals: dict[str, float]) -> str:
        bits = []
        for c in cats:
            if c.key == "save_pct":
                sa = totals.get("shots_against", 0.0)
                if sa:
                    bits.append(f"{c.label} {totals.get('saves', 0.0) / sa:.3f}")
            elif c.key in totals:
                bits.append(f"{c.label} {totals[c.key]:g}")
        return "  ".join(bits) or "-"

    head = f"#{line.proposal_id} {line.add_name}" + (
        f" for {line.drop_name}" if line.drop_name else " (added into an open spot)"
    )
    out = [f"{head}, from {line.made_on}: {line.nights} night(s) in the lineup"]
    out.append(f"    {line.add_name:22s} {fmt(line.add_totals)}")
    if line.drop_name:
        out.append(
            f"    {line.drop_name:22s} {fmt(line.drop_totals)}"
            f"  ({line.drop_nights} of those nights played)"
        )
    return out
