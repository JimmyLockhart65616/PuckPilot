"""The week as a plan: what the moves go after, what they cost, what is given up.

The league scores every category as its own win, loss or tie, so the number a
week is played for is the expected count of categories won - additive, one
category at a time. A plan for that is not a list of categories marked "go
after": early in a week nearly all of them are between 15% and 85%, and a card
that calls ten of eleven "live" says nothing.

What a person needs to read instead, all from the same calibrated odds:

    go after       the categories the proposed moves raise - where this week's
                   acquisitions are being spent
    paying for it  the categories those same moves lower: the price of the
                   ones above, named rather than buried in each card
    giving up      long shots the plan spends nothing on, with what every
                   acquisition left could do for each one alone
    toss-ups       undecided, and nothing proposed moves them
    safe           likely already; nothing to spend

The 15%/85% marks (`week.LIKELY`) only decide where a category is listed. No
decision reads them: the moves are priced by expected categories, which spends
next to nothing on a long shot without being told to.

Nothing here is decided or approved. It is recomputed on every run from that
run's odds - Yahoo's banked totals plus the days left - so it moves as the week
is played.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from puckpilot.engine.categories import Category

TARGET = "target"
COST = "cost"
GIVE_UP = "give_up"
SAFE = "safe"
TOSS_UP = "toss_up"

# A change in a category's chance smaller than this is not what the moves are
# for, nor a price worth naming.
MOVED = 0.03

TITLES = {
    TARGET: "Go after",
    COST: "Paying for it",
    GIVE_UP: "Giving up",
    TOSS_UP: "Toss-ups",
    SAFE: "Safe",
}
ORDER = (TARGET, COST, GIVE_UP, TOSS_UP, SAFE)


@dataclass(frozen=True)
class Row:
    category: Category
    now: float  # expected score as the roster stands: P(win) + half P(tie)
    then: float | None  # with every proposed move made; None with none
    ceiling: float | None  # every acquisition left spent on this one alone
    role: str

    @property
    def change(self) -> float:
        return 0.0 if self.then is None else self.then - self.now


@dataclass(frozen=True)
class GamePlan:
    week: int
    opponent: str
    rows: tuple[Row, ...]
    expected: float
    planned: float | None
    moves: int
    adds_left: int | None
    starts: tuple[int, int]
    banked: bool
    days_left: int
    bench_calls: dict[str, int]
    lineup_by: str
    extra_game: tuple[str, ...] = ()

    @classmethod
    def from_week(cls, plan, lineup_by: str = "season value") -> GamePlan | None:
        """The plan for a `week.WeekPlan`, or None when it carries no odds."""
        from puckpilot.season.week import LIKELY

        odds = getattr(plan, "odds", None)
        if odds is None:
            return None
        planned = getattr(plan, "planned", None)
        ceiling = getattr(plan, "ceiling", {}) or {}
        rows = []
        for c in odds.cats:
            now = c.expected
            after = planned.of(c.category.key) if planned is not None else None
            then = after.expected if after is not None else None
            change = 0.0 if then is None else then - now
            if change >= MOVED:
                role = TARGET
            elif change <= -MOVED:
                role = COST
            elif now <= 1.0 - LIKELY:
                role = GIVE_UP
            elif now >= LIKELY:
                role = SAFE
            else:
                role = TOSS_UP
            rows.append(
                Row(
                    category=c.category,
                    now=now,
                    then=then,
                    ceiling=ceiling.get(c.category.key),
                    role=role,
                )
            )
        return cls(
            week=plan.week,
            opponent=getattr(plan, "opponent", ""),
            rows=tuple(rows),
            expected=odds.expected,
            planned=planned.expected if planned is not None else None,
            moves=len(getattr(plan, "targets", ())) if planned is not None else 0,
            adds_left=getattr(plan, "adds_left_week", None),
            starts=(getattr(plan, "our_games", 0), getattr(plan, "their_games", 0)),
            banked=bool(getattr(plan, "banked", False)),
            days_left=getattr(plan, "days_left", 0),
            bench_calls=dict(getattr(plan, "bench_calls", {}) or {}),
            lineup_by=lineup_by,
            extra_game=_extra_game(getattr(plan, "extra_game", {}) or {}),
        )

    def of(self, role: str) -> tuple[Row, ...]:
        return tuple(r for r in self.rows if r.role == role)

    # -- in words -------------------------------------------------------------

    def title(self) -> str:
        return f"Week {self.week}" + (f" vs {self.opponent}" if self.opponent else "")

    def head(self) -> str:
        n = len(self.rows)
        line = f"Expect {self.expected:.1f} of {n}"
        if self.days_left == 0:
            return line + " - the week is over"
        if self.planned is not None and self.moves:
            the = (
                "the proposed move is"
                if self.moves == 1
                else f"the {self.moves} proposed moves are"
            )
            return line + f" -> {self.planned:.1f} if {the} made"
        if self.adds_left == 0:
            return line + " - no acquisitions left this week"
        return line + " - no pickup clears the bar this week"

    def groups(self, sep: str = " · ") -> list[tuple[str, list[str]]]:
        """(title, lines) in reading order. `sep` joins a one-line group: the
        page's middle dot, or plain ASCII where a console cannot print it."""
        out: list[tuple[str, list[str]]] = []
        for role in ORDER:
            rows = list(self.of(role))
            if not rows:
                continue
            if role in (TARGET, COST):
                rows.sort(key=lambda r: -abs(r.change))
                lines = [f"{r.category.label} {_pct(r.now)} -> {_pct(r.then)}" for r in rows]
            elif role == GIVE_UP:
                rows.sort(key=lambda r: r.now)
                lines = [self._give_up_line(r) for r in rows]
            else:
                rows.sort(key=lambda r: -r.now)
                lines = [sep.join(f"{r.category.label} {_pct(r.now)}" for r in rows)]
            out.append((TITLES[role], lines))
        if not self.moves and self.extra_game and self.days_left:
            out.insert(0, ("An extra skater game counts most in", [", ".join(self.extra_game)]))
        return out

    def _give_up_line(self, r: Row) -> str:
        line = f"{r.category.label} {_pct(r.now)}"
        if self.adds_left == 0:
            return line + " - no acquisitions left to change it"
        if r.ceiling is not None and self.adds_left:
            n = self.adds_left
            spent = "the acquisition left" if n == 1 else f"all {n} acquisitions left"
            return line + f" - {spent}, on it alone, would reach at most {_pct(r.ceiling)}"
        return line

    def notes(self, sep: str = " · ") -> list[str]:
        out = []
        if self.adds_left is not None:
            levers = f"{self.adds_left} acquisition(s) left this week"
            if self.moves:
                levers += f", the plan uses {self.moves}"
        else:
            levers = "no weekly acquisition limit"
        left = " left" if self.banked else ""
        out.append(f"{levers}{sep}starts{left} {self.starts[0]} v {self.starts[1]}")
        out.append(self._lineup_line())
        return out

    def _lineup_line(self) -> str:
        total = sum(self.bench_calls.values())
        if not total:
            return "Lineup: no bench calls - everyone with a game starts"
        days = ", ".join(_weekday(d) for d in sorted(self.bench_calls))
        calls = "1 bench call" if total == 1 else f"{total} bench calls"
        return f"Lineup: {calls} ({days}), decided by {self.lineup_by}"

    def payload(self) -> dict:
        """For the page, which formats nothing: every string is final here."""
        return {
            "title": self.title(),
            "head": self.head(),
            "groups": [{"title": t, "lines": lines} for t, lines in self.groups()],
            "notes": self.notes(),
        }

    def text(self) -> str:
        lines = [f"Plan for {self.title()}: {self.head()}"]
        for title, got in self.groups(sep=", "):
            lines.append(f"  {title}:")
            lines.extend(f"    {x}" for x in got)
        lines.extend(f"  {x}" for x in self.notes(sep=", "))
        return "\n".join(lines)


def _pct(p: float | None) -> str:
    return "-" if p is None else f"{100 * p:.0f}%"


def _weekday(day: str) -> str:
    try:
        return date.fromisoformat(day).strftime("%a")
    except ValueError:
        return day


def _extra_game(by_label: dict[str, float], limit: int = 3) -> tuple[str, ...]:
    """The categories one more typical game moves most, most first."""
    best = sorted(((v, k) for k, v in by_label.items() if v > 0.005), reverse=True)[:limit]
    return tuple(k for _, k in best)
