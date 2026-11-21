"""Does the replay's league look like the live one?

The add gate judges every search knob in a replayed league: bots draft it,
and its rivals are frozen, or stream on each week's first morning. A knob that
pays there pays in that league. The live one, seven weeks in, held three to
four goalies a team and made dozens of mid-week goalie adds, while the
replay's draft-bot rivals held what the draft's cap allowed and made none -
so every goalie knob was judged against rivals who did not compete for
goalies at all.

`live_profile` reads the same per-team-week numbers the gate keeps for its
replay (`AddGateReport.profile`) from the live league's tables, and `text`
sets them side by side: mean, and the 10th / 90th percentiles.
"""

from __future__ import annotations

import json
import sqlite3

KEYS = ("goalies held", "goalie appearances", "saves", "wins", "goalie adds")


def live_profile(conn: sqlite3.Connection, runtime) -> dict[str, list[float]]:
    """key -> one value per final team-week of the live league.

    Goalie appearances are not in Yahoo's week stats, so the live side has
    none; the rest are Yahoo's own numbers (saves, wins), the week-start
    roster (goalies held) and the transactions (goalie adds).
    """
    label_of = runtime.stat_labels()
    weeks = {w.number: w for w in runtime.weeks}
    out: dict[str, list[float]] = {k: [] for k in KEYS}
    rows = conn.execute(
        "SELECT week, team_key, stats_json FROM league_week_stats WHERE league_key = ? "
        "AND status = 'postevent' ORDER BY week, team_key",
        (runtime.league_key,),
    ).fetchall()
    for week, team, stats in rows:
        w = weeks.get(int(week))
        if w is None:
            continue
        vals = {label_of.get(int(k), k): v for k, v in json.loads(stats).items() if k.isdigit()}
        held = conn.execute(
            "SELECT COUNT(*) FROM yahoo_roster_snapshots WHERE team_key = ? AND date = ? "
            "AND ',' || eligible || ',' LIKE '%,G,%'",
            (team, w.start),
        ).fetchone()[0]
        adds = conn.execute(
            "SELECT COUNT(*) FROM league_transactions WHERE league_key = ? AND team_key = ? "
            "AND action = 'add' AND positions LIKE '%G%' AND substr(at, 1, 10) BETWEEN ? AND ?",
            (runtime.league_key, team, w.start, w.end),
        ).fetchone()[0]
        if held:
            out["goalies held"].append(float(held))
        for key, label in (("saves", "SV"), ("wins", "W")):
            if vals.get(label) is not None:
                out[key].append(float(vals[label]))
        out["goalie adds"].append(float(adds))
    return out


def _summary(xs: list[float]) -> str:
    if not xs:
        return f"{'-':>20}"
    s = sorted(xs)
    lo, hi = s[int(0.1 * (len(s) - 1))], s[int(0.9 * (len(s) - 1))]
    return f"{sum(s) / len(s):7.2f} ({lo:5.1f}-{hi:5.1f})"


def text(replay: dict[str, list[float]], live: dict[str, list[float]], season: str) -> str:
    out = [
        f"Realism - the replayed {season} league v this live league, per team-week:",
        f"  {'':20}{'replay mean (p10-p90)':>22}  {'live mean (p10-p90)':>22}",
    ]
    for k in KEYS:
        out.append(f"  {k:20}{_summary(replay.get(k, [])):>22}  {_summary(live.get(k, [])):>22}")
    n_r = len(replay.get("goalie adds", []))
    n_l = len(live.get("goalie adds", []))
    out.append(f"  ({n_r} replay team-weeks, {n_l} live; live appearances are not reported)")
    return "\n".join(out)
