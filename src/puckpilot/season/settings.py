"""What the league itself says its rules are, read from Yahoo rather than typed.

`leagues/*.toml` describes the league as a human transcribed it. That is right
for things Yahoo does not know - this league tracks keepers by hand, so nothing
else can - and wrong for everything Yahoo does know, because a transcription is
a copy that silently goes stale.

Everything here is Yahoo's own answer: the week calendar, the lineup lock mode,
the waiver rules, the acquisition caps, the roster slots. A league with twelve
categories and a 22-week season configures itself, and so does one with none of
that, which is the point.

The week calendar is fetched, never computed. Yahoo says this season runs
2026-09-29 to 2027-03-28 across 25 weeks; Monday-Sunday arithmetic from the
start date lands on 2027-03-21, a week short, because at least one week is
longer than seven days. Deriving week boundaries would therefore have put every
"games this week" count into the wrong bucket for the back half of the season -
the same failure mode as assuming keepers filled a seat's earliest rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from puckpilot.engine.valuation import LeagueShape

# Yahoo's slot names vs the single-letter positions the engines use. Util is
# handled separately (LeagueShape carries it as a count), and the rest are not
# startable.
YAHOO_TO_POS = {"C": "C", "LW": "L", "RW": "R", "D": "D", "G": "G"}
BENCH_SLOTS = {"BN"}
IR_SLOTS = {"IR", "IR+", "IR-LT", "NA"}
UTIL_SLOTS = {"Util", "UTIL", "W", "F"}

# Statuses that mean the player cannot play tonight no matter what the schedule
# says. DTD is deliberately absent: day-to-day players play most nights, and
# treating them as out would bench half a roster every February.
OUT_STATUSES = {"IR", "IR-LT", "IR-R", "O", "NA", "SUSP"}


class SettingsError(RuntimeError):
    """The league's settings cannot answer the question asked of them."""


@dataclass(frozen=True)
class RosterSlot:
    position: str
    count: int
    starting: bool


@dataclass(frozen=True)
class Week:
    number: int
    start: str
    end: str

    def dates(self) -> list[str]:
        a, b = date.fromisoformat(self.start), date.fromisoformat(self.end)
        return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]

    def contains(self, day: str) -> bool:
        return self.start <= day <= self.end


@dataclass(frozen=True)
class LeagueRuntime:
    """Yahoo's own description of the league, as of when it was fetched."""

    league_key: str
    name: str
    num_teams: int
    scoring_type: str
    yahoo_season: str
    start_date: str
    end_date: str
    start_week: int
    end_week: int
    current_week: int
    current_date: str
    playoff_start_week: int
    num_playoff_teams: int
    weekly_deadline: str
    roster_type: str
    waiver_type: str
    waiver_rule: str
    waiver_days: int
    uses_faab: bool
    max_adds: int | None
    max_weekly_adds: int | None
    min_games_played: int
    trade_end_date: str
    slots: tuple[RosterSlot, ...]
    weeks: tuple[Week, ...] = ()
    fetched_at: str = ""

    # -- calendar ----------------------------------------------------------

    @property
    def nhl_season(self) -> str:
        """Yahoo's "2026" as the eight-digit form the rest of the codebase uses."""
        y = int(self.yahoo_season)
        return f"{y}{y + 1}"

    @property
    def regular_weeks(self) -> int:
        return self.playoff_start_week - self.start_week

    def week_of(self, day: str) -> int:
        """The fantasy week containing `day`.

        Raises rather than guessing when the calendar was never fetched: a wrong
        week silently mis-counts every "games remaining" number downstream.
        """
        if not self.weeks:
            raise SettingsError(
                "week calendar not loaded - run `ppilot season settings --refresh`. "
                "Week boundaries are not uniform and must not be computed."
            )
        for w in self.weeks:
            if w.contains(day):
                return w.number
        raise SettingsError(f"{day} falls outside weeks {self.start_week}-{self.end_week}")

    def week(self, number: int) -> Week:
        for w in self.weeks:
            if w.number == number:
                return w
        raise SettingsError(f"week {number} not in the loaded calendar")

    # -- roster ------------------------------------------------------------

    @property
    def is_daily_lineup(self) -> bool:
        """True when lineups lock per game rather than once a week."""
        return self.weekly_deadline == "intraday" or self.roster_type == "date"

    @property
    def bench_slots(self) -> int:
        return sum(s.count for s in self.slots if s.position in BENCH_SLOTS)

    @property
    def ir_slots(self) -> int:
        return sum(s.count for s in self.slots if s.position in IR_SLOTS)

    @property
    def util_slots(self) -> int:
        return sum(s.count for s in self.slots if s.position in UTIL_SLOTS)

    def shape(self) -> LeagueShape:
        """The `LeagueShape` the valuation and lineup engines already take."""
        starting = [
            (YAHOO_TO_POS[s.position], s.count)
            for s in self.slots
            if s.starting and s.position in YAHOO_TO_POS
        ]
        if not starting:
            raise SettingsError(f"no startable roster slots in {self.league_key}")
        return LeagueShape(
            n_teams=self.num_teams,
            slots=tuple(starting),
            util_slots=self.util_slots,
            bench_slots=self.bench_slots,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_payload(
        cls, flat: dict, weeks: tuple[Week, ...] = (), fetched_at: str = ""
    ) -> LeagueRuntime:
        """Build from a `flatten()`ed `/league/{key}/settings` response."""

        def need(key: str) -> str:
            v = flat.get(key)
            if v in (None, ""):
                raise SettingsError(f"Yahoo settings are missing {key!r}")
            return str(v)

        def num(key: str, default: int | None = None) -> int:
            v = flat.get(key)
            if v in (None, ""):
                if default is None:
                    raise SettingsError(f"Yahoo settings are missing {key!r}")
                return default
            return int(v)

        slots = tuple(
            RosterSlot(
                position=str(rp["position"]),
                count=int(rp.get("count", 0)),
                starting=bool(int(rp.get("is_starting_position", 0))),
            )
            for entry in flat.get("roster_positions", [])
            if (rp := entry.get("roster_position") if isinstance(entry, dict) else None)
        )
        if not slots:
            raise SettingsError("Yahoo settings carried no roster_positions")

        # max_adds is absent in leagues that do not cap acquisitions; that is a
        # real setting ("unlimited"), not a missing field, so it stays None and
        # waivers.budget_threshold already treats None that way.
        return cls(
            league_key=need("league_key"),
            name=need("name"),
            num_teams=num("num_teams"),
            scoring_type=need("scoring_type"),
            yahoo_season=need("season"),
            start_date=need("start_date"),
            end_date=need("end_date"),
            start_week=num("start_week"),
            end_week=num("end_week"),
            current_week=num("current_week"),
            current_date=need("current_date"),
            playoff_start_week=num("playoff_start_week"),
            num_playoff_teams=num("num_playoff_teams", 0),
            weekly_deadline=str(flat.get("weekly_deadline", "")),
            roster_type=str(flat.get("roster_type", "")),
            waiver_type=str(flat.get("waiver_type", "")),
            waiver_rule=str(flat.get("waiver_rule", "")),
            waiver_days=num("waiver_time", 0),
            uses_faab=bool(int(flat.get("uses_faab", 0) or 0)),
            max_adds=int(flat["max_adds"]) if flat.get("max_adds") else None,
            max_weekly_adds=int(flat["max_weekly_adds"]) if flat.get("max_weekly_adds") else None,
            min_games_played=num("min_games_played", 0),
            trade_end_date=str(flat.get("trade_end_date", "")),
            slots=slots,
            weeks=weeks,
            fetched_at=fetched_at,
        )
