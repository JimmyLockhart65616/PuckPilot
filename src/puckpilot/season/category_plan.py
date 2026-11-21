"""The season's category plan: where each category stands, and the stance on it.

A week's odds say what to chase this week. Over a season the question is
different: which categories are worth a roster spot at all, given every week
left and every rival. Seven weeks into 2026-27 the roster had drifted into
winning hits by twenty-five while conceding assists, power-play points and all
three goalie categories - and nothing on the page said so.

Per category, from the league's own weekly results (`league_week_stats`):
- the record against the opponents actually played;
- the all-play share: of every other team's week, how many our week beat -
  what the category is like independent of the schedule;
- its season leverage (`season.outlook`): what one more unit a week is worth,
  against every rival's current roster, restated as what one typical
  starter's worth of it buys over the weeks left;
- a proposed stance: chase, hold or concede.

A manager may pin a stance. A pin weights the category's season leverage
(`TILT`: chase 1.5, hold 1, concede 0) - so it acts through the season term
only (`authority.transactions.season_weight`), never through this week's odds.
With nothing pinned, nothing changes: the season term is the measured one.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class Stance(StrEnum):
    CHASE = "chase"
    HOLD = "hold"
    CONCEDE = "concede"


# How a pinned stance weights a category's season leverage.
TILT = {Stance.CHASE: 1.5, Stance.HOLD: 1.0, Stance.CONCEDE: 0.0}

# The proposal: by what a typical starter's worth buys over the season,
# the top third of categories are chase and the bottom third concede when the
# all-play share is under this, else hold. Information only - a proposal
# changes nothing until a manager pins it.
CONCEDE_BELOW = 0.35


@dataclass(frozen=True)
class CategoryLine:
    key: str
    label: str
    won: int
    lost: int
    tied: int
    all_play: float | None  # share of the league's other team-weeks beaten
    leverage: float | None  # expected categories per unit more a week
    starter_week: float | None  # a typical starter's week of it
    weeks_left: int
    proposed: Stance
    pinned: Stance | None

    @property
    def per_spot(self) -> float | None:
        """Categories over the season a typical starter's worth of it buys."""
        if self.leverage is None or self.starter_week is None:
            return None
        return self.leverage * self.starter_week * self.weeks_left

    @property
    def stance(self) -> Stance:
        return self.pinned or self.proposed


def _outcome(higher_is_better: bool, a, b) -> float | None:
    if a is None or b is None:
        return None
    if abs(a - b) <= 1e-9:
        return 0.5
    return 1.0 if (a > b) == higher_is_better else 0.0


def records(conn, runtime, league, team_key: str) -> dict[str, tuple[int, int, int, float | None]]:
    """category key -> (won, lost, tied, all-play share) over the final weeks."""
    label_of = runtime.stat_labels()
    by_week: dict[int, dict[str, tuple[str, dict]]] = {}
    for week, team, opp, stats in conn.execute(
        "SELECT week, team_key, opponent_key, stats_json FROM league_week_stats "
        "WHERE league_key = ? AND status = 'postevent'",
        (runtime.league_key,),
    ):
        vals = {label_of.get(int(k), k): v for k, v in json.loads(stats).items() if k.isdigit()}
        by_week.setdefault(int(week), {})[team] = (opp, vals)
    out: dict[str, tuple[int, int, int, float | None]] = {}
    for c in league.all_cats:
        won = lost = tied = 0
        beat = played = 0.0
        for teams in by_week.values():
            if team_key not in teams:
                continue
            opp, ours = teams[team_key]
            mine = ours.get(c.label)
            r = _outcome(c.higher_is_better, mine, teams.get(opp, ("", {}))[1].get(c.label))
            if r is not None:
                won += r == 1.0
                lost += r == 0.0
                tied += r == 0.5
            for other, (_o, vals) in teams.items():
                x = _outcome(c.higher_is_better, mine, vals.get(c.label))
                if other != team_key and x is not None:
                    beat += x
                    played += 1
        out[c.key] = (won, lost, tied, beat / played if played else None)
    return out


def starter_weeks(roster_games: dict[int, float], rates: dict[int, dict[str, float]], keys):
    """category key -> the median rostered starter's week of it (`roster_games`
    is each player's expected starts in a typical week)."""
    out: dict[str, float] = {}
    for k in keys:
        xs = sorted(
            rates.get(pid, {}).get(k, 0.0) * n
            for pid, n in roster_games.items()
            if n > 0 and rates.get(pid, {}).get(k, 0.0) > 0
        )
        if xs:
            out[k] = xs[len(xs) // 2]
    return out


def propose(lines: list[CategoryLine]) -> dict[str, Stance]:
    """The tool's stance per category: see `CONCEDE_BELOW`."""
    scored = sorted(
        (ln for ln in lines if ln.per_spot is not None), key=lambda ln: -(ln.per_spot or 0.0)
    )
    n = len(scored)
    out: dict[str, Stance] = {ln.key: Stance.HOLD for ln in lines}
    for i, ln in enumerate(scored):
        if i < n / 3:
            out[ln.key] = Stance.CHASE
        elif i >= 2 * n / 3 and (ln.all_play is None or ln.all_play < CONCEDE_BELOW):
            out[ln.key] = Stance.CONCEDE
    return out


def build(
    records_by_key,
    league,
    leverage: dict[str, float] | None,
    starter: dict[str, float],
    weeks_left: int,
    pinned: dict[str, Stance],
) -> list[CategoryLine]:
    """The plan's lines, every scored category in the league's order."""
    lines = []
    for c in league.all_cats:
        won, lost, tied, share = records_by_key.get(c.key, (0, 0, 0, None))
        lines.append(
            CategoryLine(
                key=c.key,
                label=c.label,
                won=won,
                lost=lost,
                tied=tied,
                all_play=share,
                leverage=(leverage or {}).get(c.key),
                starter_week=starter.get(c.key),
                weeks_left=weeks_left,
                proposed=Stance.HOLD,
                pinned=pinned.get(c.key),
            )
        )
    proposed = propose(lines)
    return [
        CategoryLine(**{**ln.__dict__, "proposed": proposed.get(ln.key, Stance.HOLD)})
        for ln in lines
    ]


def text(lines: list[CategoryLine], weight: float) -> str:
    def pct(x):
        return "  -  " if x is None else f"{x:5.0%}"

    off = "; 0 = this week only, so stances do nothing" if not weight else ""
    out = [
        f"Category plan - {lines[0].weeks_left if lines else 0} weeks left. A stance weights the "
        "season term only",
        f"(season_weight {weight:g}{off}).",
        "",
        "  cat   record    all-play  a starter's week buys   proposed  pinned",
    ]
    for ln in lines:
        buys = "      -" if ln.per_spot is None else f"{ln.per_spot:+7.2f} cats"
        out.append(
            f"  {ln.label:5} {ln.won:2d}-{ln.lost:2d}-{ln.tied:1d}   {pct(ln.all_play)}"
            f"     {buys:>14}        {ln.proposed.value:8}  {ln.pinned.value if ln.pinned else ''}"
        )
    out += [
        "",
        "all-play: of every other team's week, the share ours beat. A starter's week buys: one",
        "more median starter's worth of the category each week, in expected categories won over",
        "the weeks left, against every rival's current roster.",
    ]
    return "\n".join(out)


# -- pins --------------------------------------------------------------------


def pins(conn: sqlite3.Connection, manager: str, league_key: str) -> dict[str, Stance]:
    """category key -> the stance this manager pinned."""
    rows = conn.execute(
        "SELECT category, stance FROM category_plan WHERE manager = ? AND league_key = ?",
        (manager, league_key),
    )
    known = {s.value for s in Stance}
    return {k: Stance(v) for k, v in rows if v in known}


def pin(
    conn: sqlite3.Connection,
    manager: str,
    league_key: str,
    category: str,
    stance: Stance | None,
    via: str = "cli",
    at: str | None = None,
) -> None:
    """Pin a stance, or with None take it off (back to the proposal)."""
    if stance is None:
        conn.execute(
            "DELETE FROM category_plan WHERE manager = ? AND league_key = ? AND category = ?",
            (manager, league_key, category),
        )
    else:
        conn.execute(
            "INSERT OR REPLACE INTO category_plan (manager, league_key, category, stance, "
            "set_at, set_via) VALUES (?, ?, ?, ?, ?, ?)",
            (
                manager,
                league_key,
                category,
                Stance(stance).value,
                at or datetime.now(UTC).isoformat(timespec="seconds"),
                via,
            ),
        )
    conn.commit()


def tilt(pinned: dict[str, Stance]) -> dict[str, float]:
    """category key -> the weight on its season leverage; unpinned: absent (1)."""
    return {k: TILT[s] for k, s in pinned.items()}
