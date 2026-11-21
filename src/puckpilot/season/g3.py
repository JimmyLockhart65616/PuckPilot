"""Gate G3 for the whole league: did each week's projection match what happened?

G1 calibrates the odds on replayed seasons; G3 is the live check, and until
2026-10-31 it could only look at one matchup in six - ours. With every team's
week-start roster and final totals on file (`league_feed`), each final week
gives twelve team-weeks: each projected exactly as the tool would have on the
week's first morning - models clamped to the day before (`AsOfValues`,
`AsOfGoalieSource`), nothing banked, the lineup the tool itself would set -
and compared with Yahoo's totals.

It answers three questions, each one a decision (STATUS.md, pre-registered):

- Are the odds calibrated on live weeks? (Brier per category; expected
  categories per team-week against categories won; reliability by band.)
- Do teams that stream goalies during a week beat their projected saves -
  the volume the projection cannot see coming?
- Is our own team projected differently from the rest of the league?

What a projection cannot know is part of the answer, not noise to remove:
real managers set imperfect lineups, add and drop mid-week, and unmapped
players (the player map misses a few) project at nothing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date as _date
from datetime import timedelta

POSITIONS = ("C", "LW", "RW", "D", "G")


@dataclass(frozen=True)
class CatCase:
    week: int
    team_key: str
    opp_key: str
    cat: str  # the league's label
    kind: str  # skater | goalie
    projected: float
    projected_opp: float
    actual: float | None
    actual_opp: float | None
    p_win: float
    p_tie: float
    outcome: float  # 1 won, 0.5 tied, 0 lost

    @property
    def expected(self) -> float:
        return self.p_win + 0.5 * self.p_tie


@dataclass
class G3Report:
    cases: list[CatCase]
    goalie_adds: dict[tuple[int, str], int] = field(default_factory=dict)
    unmapped: int = 0
    our_team: str = ""

    def brier(self, cases=None) -> float:
        cs = self.cases if cases is None else cases
        return sum((c.expected - c.outcome) ** 2 for c in cs) / len(cs) if cs else float("nan")

    def team_weeks(self) -> dict[tuple[int, str], list[CatCase]]:
        out: dict[tuple[int, str], list[CatCase]] = {}
        for c in self.cases:
            out.setdefault((c.week, c.team_key), []).append(c)
        return out

    def expected_error(self, team: str | None = None) -> tuple[float, float, int]:
        """Mean (categories won - expected) per team-week, its SE and n."""
        d = [
            sum(c.outcome for c in cs) - sum(c.expected for c in cs)
            for (_, t), cs in self.team_weeks().items()
            if team is None or t == team
        ]
        return _mean_se(d)

    def bias(self, cat: str, team: str | None = None, exclude: str | None = None):
        """Mean (actual - projected) team total in one category, SE and n."""
        d = [
            c.actual - c.projected
            for c in self.cases
            if c.cat == cat
            and c.actual is not None
            and (team is None or c.team_key == team)
            and (exclude is None or c.team_key != exclude)
        ]
        return _mean_se(d)

    def reliability(self, bands=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)) -> list[tuple]:
        out = []
        for lo, hi in zip(bands, bands[1:], strict=False):
            cs = [c for c in self.cases if lo <= c.expected < hi]
            if cs:
                out.append((lo, min(hi, 1.0), len(cs), sum(c.expected for c in cs) / len(cs),
                            sum(c.outcome for c in cs) / len(cs)))  # fmt: skip
        return out

    def streaming(self, cat: str) -> tuple[tuple, tuple]:
        """(actual - projected) in `cat` for team-weeks with a goalie added
        during the week, and for the rest."""
        streamed, rest = [], []
        for c in self.cases:
            if c.cat != cat or c.actual is None:
                continue
            (streamed if self.goalie_adds.get((c.week, c.team_key)) else rest).append(
                c.actual - c.projected
            )
        return _mean_se(streamed), _mean_se(rest)

    def text(self) -> str:
        n_tw = len(self.team_weeks())
        weeks = sorted({c.week for c in self.cases})
        lines = [
            f"G3 (live, whole league) - {n_tw} team-weeks, weeks {weeks}, "
            f"{len(self.cases)} category outcomes"
            + (
                f"; {self.unmapped} rostered players unmapped (projected at nothing)"
                if self.unmapped
                else ""
            ),  # fmt: skip
            "",
            f"Brier per category: {self.brier():.4f}   (G1 test season: 0.130)",
        ]
        m, se, n = self.expected_error()
        lines.append(f"categories won - expected, per team-week: {m:+.2f} +/- {se:.2f} (n={n})")
        if self.our_team:
            m, se, n = self.expected_error(self.our_team)
            lines.append(f"  our team: {m:+.2f} +/- {se:.2f} (n={n})")
        lines += ["", "Reliability (expected score band: n, said, happened):"]
        for lo, hi, k, said, got in self.reliability():
            lines.append(f"  {lo:.1f}-{hi:.1f}: {k:4d}  {said:.3f}  {got:.3f}")
        lines += ["", "Bias, actual - projected team total (all teams | ours | the rest):"]
        for cat in sorted({c.cat for c in self.cases}, key=_order):
            a = self.bias(cat)
            row = f"  {cat:5s} {a[0]:+8.2f} +/- {a[1]:.2f}"
            if self.our_team:
                o = self.bias(cat, team=self.our_team)
                r = self.bias(cat, exclude=self.our_team)
                row += f"   | {o[0]:+8.2f} +/- {o[1]:.2f} | {r[0]:+8.2f} +/- {r[1]:.2f}"
            lines.append(row)
        if self.goalie_adds:
            lines += ["", "Goalie volume v in-week goalie adds (actual - projected):"]
            for cat in ("SV", "W"):
                if not any(c.cat == cat for c in self.cases):
                    continue
                (sm, sse, sn), (rm, rse, rn) = self.streaming(cat)
                lines.append(
                    f"  {cat:3s} added a goalie that week {sm:+7.2f} +/- {sse:.2f} (n={sn})   "
                    f"did not {rm:+7.2f} +/- {rse:.2f} (n={rn})"
                )
        return "\n".join(lines)


def _order(label: str) -> int:
    order = ("G", "A", "P", "+/-", "PIM", "PPP", "SHP", "GWG", "SOG", "HIT", "BLK", "W", "GA",
             "GAA", "SV", "SA", "SV%", "SO")  # fmt: skip
    return order.index(label) if label in order else len(order)


def _mean_se(xs) -> tuple[float, float, int]:
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n), n


def saved_roster(conn, league_key: str, team_key: str, date: str):
    """A team's roster as saved for `date` (`yahoo_roster_snapshots`), or None."""
    from puckpilot.season.roster import RosterPlayer, TeamRoster

    rows = conn.execute(
        "SELECT player_key, nhl_player_id, name, team_abbrev, selected_slot, eligible, status, "
        "is_editable FROM yahoo_roster_snapshots WHERE team_key = ? AND date = ? "
        "GROUP BY player_key",
        (team_key, date),
    ).fetchall()
    if not rows:
        return None
    players = []
    for key, pid, name, team, slot, elig, status, editable in rows:
        eligible = frozenset(x for x in (elig or "").split(",") if x)
        primary = next((x for x in POSITIONS if x in eligible), "C")
        players.append(
            RosterPlayer(
                player_key=key,
                yahoo_id=str(key).rsplit(".", 1)[-1],
                name=name,
                team=team or "",
                primary_position=primary,
                yahoo_eligible=eligible,
                selected_slot=slot or "BN",
                nhl_player_id=pid,
                status=status or "",
                is_editable=bool(editable),
            )  # fmt: skip
        )
    return TeamRoster(league_key=league_key, team_key=team_key, date=date, players=tuple(players))


def _outcome(cat, a, b) -> float:
    if a is None and b is None:
        return 0.5
    if b is None:
        return 1.0
    if a is None:
        return 0.0
    if abs(a - b) <= 1e-9:
        return 0.5
    return 1.0 if (a > b) == cat.higher_is_better else 0.0


def goalie_adds_by_week(conn, league_key: str, weeks) -> dict[tuple[int, str], int]:
    """Goalies each team added during each week (UTC dates - a move made in the
    evening of a week's last day may land in the next)."""
    out: dict[tuple[int, str], int] = {}
    for w in weeks:
        for team, n in conn.execute(
            "SELECT team_key, COUNT(*) FROM league_transactions WHERE league_key = ? "
            "AND action = 'add' AND positions LIKE '%G%' AND substr(at, 1, 10) BETWEEN ? AND ? "
            "GROUP BY team_key",
            (league_key, w.start, w.end),
        ):
            out[(w.number, team)] = n
    return out


def build(conn, runtime, league, our_team: str = "", weeks=None, progress=None) -> G3Report:
    """Every final team-week projected as of its first morning, against the result."""
    from puckpilot.draft.sim import build_universe
    from puckpilot.season.add_gate import AsOfValues
    from puckpilot.season.goalies import AsOfGoalieSource, TrailingStartShareSource
    from puckpilot.season.odds import OddsModel
    from puckpilot.season.run import _train_seasons
    from puckpilot.season.values import build_value_model
    from puckpilot.season.week import build_week_plan

    lk = runtime.league_key
    season = runtime.nhl_season
    train = _train_seasons(season)
    values = build_value_model(conn, season, train, league)
    source = TrailingStartShareSource(conn, season, fallback_season=train[0])
    frame = build_universe(conn, season, train, league).frame
    label_of = runtime.stat_labels()
    final = {
        int(r[0])
        for r in conn.execute(
            "SELECT week FROM league_week_stats WHERE league_key = ? AND status = 'postevent' "
            "GROUP BY week",
            (lk,),
        )
    }
    wanted = final if weeks is None else final & set(weeks)
    chosen = [w for w in runtime.weeks if w.number in wanted]
    cases: list[CatCase] = []
    unmapped = 0
    for w in chosen:
        cutoff = (_date.fromisoformat(w.start) - timedelta(days=1)).isoformat()
        vals, goalies = AsOfValues(values, cutoff), AsOfGoalieSource(source, cutoff)
        rows = {
            r[0]: (r[1], r[2], json.loads(r[3]))
            for r in conn.execute(
                "SELECT team_key, opponent_key, name, stats_json FROM league_week_stats "
                "WHERE league_key = ? AND week = ?",
                (lk, w.number),
            )
        }
        for team, (opp, _name, stats) in sorted(rows.items()):
            ours = saved_roster(conn, lk, team, w.start)
            theirs = saved_roster(conn, lk, opp, w.start)
            if ours is None or theirs is None or opp not in rows:
                continue
            unmapped += sum(1 for p in ours.players if p.nhl_player_id is None)
            plan = build_week_plan(
                conn, runtime, league, w, rows[opp][1], ours, theirs, [], frame, goalies, vals,
                find_targets=False, measure_room=False, odds_model=OddsModel(),
                add_scoring="odds", from_day=w.start,
            )  # fmt: skip
            opp_stats = rows[opp][2]
            by_label = {label_of.get(int(k), k): v for k, v in stats.items() if k.isdigit()}
            opp_by_label = {label_of.get(int(k), k): v for k, v in opp_stats.items()
                            if k.isdigit()}  # fmt: skip
            for o in plan.outlook:
                if o.p_win is None:
                    continue
                a = by_label.get(o.category.label)
                b = opp_by_label.get(o.category.label)
                cases.append(
                    CatCase(
                        week=w.number,
                        team_key=team,
                        opp_key=opp,
                        cat=o.category.label,
                        kind=o.category.kind,
                        projected=float(o.ours),
                        projected_opp=float(o.theirs),
                        actual=a,
                        actual_opp=b,
                        p_win=float(o.p_win),
                        p_tie=float(o.p_tie or 0.0),
                        outcome=_outcome(o.category, a, b),
                    )  # fmt: skip
                )
            if progress:
                progress(f"  week {w.number} {team}: {plan.expected:.2f} expected")
    return G3Report(cases, goalie_adds_by_week(conn, lk, chosen), unmapped, our_team)
