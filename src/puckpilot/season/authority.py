"""What the tool may do on its own, and what it must ask about first.

The split is deliberate and it is not symmetric.

A lineup change is reversible, expires in a day, costs nothing to get wrong
beyond one night of one player, and has to happen ~185 times a season at an
hour when a person is usually busy. So it runs under *standing authority*: a
written, versioned set of criteria, and anything inside them is done without
asking. The criteria live in config precisely so they can be argued with after
the fact against the audit log rather than debated in advance.

A transaction is none of those things. It spends one of a fixed number of
weekly acquisitions, it drops a player who may not come back, and in a rolling
waiver league it can be won by somebody else while you sleep. So it is
*approval-gated*, and that is enforced structurally rather than by a setting:
`TransactionAuthority` has no `enabled` flag, and the executor takes an
approved `waiver_proposals` id it cannot mint itself. Turning transactions
autonomous is not a config change; it would require deleting the mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# How to treat a player Yahoo flags but does not rule out (DTD and friends).
QUESTIONABLE_MODES = ("never", "only_if_needed", "always")
# How an add is priced (see TransactionAuthority.add_scoring).
ADD_SCORING = ("share", "odds")
# How the add search shortlists candidates (TransactionAuthority.screen).
SCREENS = ("value", "union")
# How a night with more players than slots is decided (LineupAuthority.steer).
STEER_MODES = ("off", "odds")


class AuthorityError(ValueError):
    """The authority config is not a thing the tool can act on."""


@dataclass(frozen=True)
class LineupAuthority:
    """Standing permission to change today's lineup, and its bounds."""

    enabled: bool = False
    # Leave a lineup alone unless the swap is worth this much, in the same
    # z-like per-day units `GameValueModel` produces. Measured over a real
    # season this knob barely matters in either direction - raising it to 1.0
    # changes 530 lineup moves instead of 536 and costs 0.13% of season value -
    # because daily churn comes from who has a game, not from marginal swaps.
    # Kept as a small floor against a pointless swap between near-equals.
    min_gain: float = 0.15
    # Never start a goalie less likely than this to actually start. A goalie who
    # does not play scores zero in every goalie category.
    min_goalie_p_start: float = 0.5
    # ...except when a G slot would otherwise sit empty: a goalie who does not
    # start then scores nothing, the same as the empty slot, so the floor costs
    # something and protects nothing. The floor still decides between goalies
    # competing for a slot - a 45% goalie never takes one from a 60% one here.
    fill_empty_goalie_slot: bool = False
    # "never" benches anyone carrying a status; "always" ignores the flag;
    # "only_if_needed" starts him when the alternative is an empty slot.
    start_questionable: str = "only_if_needed"
    # Yahoo's own per-week minimum (min_games_played) is a rule, not a
    # preference - falling short forfeits the category.
    enforce_min_games: bool = True
    # There is deliberately no cap on the number of changes. Lineup moves are
    # free and unlimited until a player's own game starts, so a busy night is a
    # busy schedule, not a malfunction - an earlier version capped them and
    # would have refused to act on 17.6% of days for no reason. What actually
    # bounds the day is the lock, which is a clock, not a count.
    # Names, because a person thinks in names. Config, never code.
    never_bench: tuple[str, ...] = ()
    # Move a player Yahoo lists as out, and eligible for a free IR slot, into
    # it. That frees an active roster spot, which is worth something only once
    # an add fills it - and the add is still a proposal. Off unless agreed:
    # it is a roster move a person may want to make themselves.
    manage_ir: bool = False
    # Whether an approved week protocol also re-weights the daily lineup.
    #
    # OFF, on measurement. Scored by categories won per week - the only metric
    # that can judge a mechanism whose whole purpose is trading value for
    # category wins - it came out positive in 3 of 6 runs across two seasons
    # and three seeds, with every effect inside +-0.08 categories of ~5.8.
    # That is noise, and it agrees with why: lineup headroom is zero most
    # weeks, because a roster of seventeen into thirteen slots has no choice
    # to make on a night when nine players have games. A weight cannot change
    # a decision that was never open.
    #
    # The protocol is analysis, not a switch: it says which categories are gone
    # and which are live, from the same odds the add search prices by. Gate G2
    # also found no gain in choosing goalies by those odds, so nothing in the
    # daily lineup acts on it. This flag only governs the lineup.
    follow_protocol: bool = False
    # How a night with more players than slots is decided. "off": by each
    # player's per-game value, as always. "odds": by what his game adds to
    # this week's expected categories won - from the odds rebuilt on that run,
    # Yahoo's banked score plus the days left - and tonight's goalies by
    # `odds.choose_goalies`.
    #
    # OFF, on measurement (add gate `odds-daily-h-f25-x1-k1-r-t` against the
    # same arm unsteered, 12 teams x 22 weeks, 4 drafts x 2 seasons): ahead in
    # 6 of 8 leagues, pooled +0.015 +/- 0.009 categories a week - inside noise,
    # and short of the rule set before running it (clear of 2 SE in both
    # seasons of the first draft: +0.081 +/- 0.026, +0.006 +/- 0.024). It
    # changes 14-20 nights a team-season; there are not enough bench calls for
    # the odds to matter. The fixed-weight protocol measured the same way.
    steer: str = "off"

    def __post_init__(self) -> None:
        if self.steer not in STEER_MODES:
            raise AuthorityError(
                f"authority.lineup.steer must be one of {', '.join(STEER_MODES)}; "
                f"got {self.steer!r}"
            )
        if self.start_questionable not in QUESTIONABLE_MODES:
            raise AuthorityError(
                f"authority.lineup.start_questionable must be one of "
                f"{', '.join(QUESTIONABLE_MODES)}; got {self.start_questionable!r}"
            )
        if not 0.0 <= self.min_goalie_p_start <= 1.0:
            raise AuthorityError("authority.lineup.min_goalie_p_start must be between 0 and 1")

    def describe(self) -> list[str]:
        """The criteria in words, for printing before acting under them."""
        if not self.enabled:
            return ["Lineup changes: RECOMMEND ONLY (no standing authority granted)."]
        return [
            "Lineup changes: AUTONOMOUS within these criteria -",
            f"  swap only when it gains at least {self.min_gain:.2f} for the day",
            f"  never start a goalie below {self.min_goalie_p_start:.0%} to start"
            + (
                " - unless a G slot would otherwise be empty" if self.fill_empty_goalie_slot else ""
            ),
            f"  players flagged day-to-day: {self.start_questionable.replace('_', ' ')}",
            f"  weekly goalie minimum enforced: {'yes' if self.enforce_min_games else 'no'}",
            "  out players moved to a free IR slot: "
            + ("yes" if self.manage_ir else "no (recommended, not made)"),
            "  changes are unlimited until each player's game starts",
            "  week protocol steers the lineup: "
            + ("yes" if self.follow_protocol else "no (measured inert)"),
            "  bench calls decided by: "
            + ("this week's odds" if self.steer == "odds" else "season value"),
            *([f"  never benched: {', '.join(self.never_bench)}"] if self.never_bench else []),
        ]


@dataclass(frozen=True)
class TransactionAuthority:
    """Bounds on what may be *proposed*. There is deliberately no `enabled`.

    Adds, drops and claims are never executed without an approved proposal row,
    so this class tunes what reaches a person, not what happens without one.
    """

    # How much of a live category gap a swap must close to be worth raising.
    # 0.25 is a quarter of it. A proposal the person will reject is a
    # notification that trains them to ignore the next one.
    #
    # This is a SHARE, not a score. It used to be an abstract value number, and
    # when adds started being priced by re-slotting the week the same 0.5 quietly
    # became "must close half a gap" and cut five proposals to one.
    min_weekly_gain: float = 0.25
    # More pending proposals than this means the tool is guessing, not advising.
    max_pending: int = 5
    # How an add is priced: "share" of each live gap it closes (the original),
    # or "odds" - the change in expected categories won this week, from the
    # calibrated model. Gate G2 (`ppilot season add-gate`, 12 teams x 22 weeks,
    # rivals frozen): odds-weekly against share-weekly +0.15 +/- 0.09 (2025-26)
    # and +0.02 +/- 0.11 (2024-25) categories a week - never worse, not clear of
    # noise in either season, which was the bar set before running it - while
    # making about 30% fewer adds (504 v 727, 539 v 744). So "share" stays the
    # default and "odds" is a manager's choice, with that on the record.
    # Replicated afterwards on three more drafts (seeds 7, 42, 99) x both
    # seasons: all eight leagues favour odds, pooled +0.22 +/- 0.03 categories
    # a week (4 of 8 clear 2 SE alone), ~28% fewer adds in every one. The
    # default waits on the manager; the evidence says switch.
    add_scoring: str = "share"
    # Under "odds", the least an add must raise expected categories won.
    min_expected_gain: float = 0.1
    # Acquisitions kept back for the playoffs: once the season's remaining
    # count falls to this, the regular season stops proposing.
    playoff_reserve: int = 6
    # Roster spots that rotate: only the players worth least over the rest of
    # the season may be dropped for a streamer (an upgrade can replace anyone).
    # Measured 2 v 3 v 4 after week 1 (add gate, both seasons): neither 3 nor 4
    # was positive in both and clear of noise, so it stays 2.
    stream_spots: int = 2
    # The last days of a week on which an add still waiting for a decision must
    # pay over the rest of this week AND the next week together, or it is
    # withdrawn (it never adds a move). 1 = Sunday, 0 = never. Asked for on
    # 2026-10-04, after two adds approved for a lost week's last two days
    # turned out to cost categories the week after (-0.32, -0.12). Judgement,
    # not measurement: the add gate replays no waiting proposals to test it on.
    look_ahead_days: int = 1
    # The last days of a week on which the SEARCH also prices against the next
    # week, spending this week's leftover acquisitions. Measured in the add
    # gate (12 teams x 22 weeks) against the plain weekly search: +0.05 +/-
    # 0.07 (2024-25) and -0.33 +/- 0.07 (2025-26) categories a week, with ~20%
    # more adds. Queued until the rosters unlock (a drop who plays that day
    # keeps his game) it was still -0.04 +/- 0.09 and -0.28 +/- 0.08: the cost
    # is the churn, not the lost Sunday game. Off by default.
    preload_days: int = 0
    # A search on every morning of the week but its first (Monday's weekly
    # search) and the look-ahead days: a swap that gets more of the rest of
    # the week's empty slots filled, made mid-week rather than waiting for
    # Monday - asked for after week 1. None = off: mid-week, the search runs
    # only when a roster spot opens. A mid-week move must clear this many
    # expected categories, not `min_expected_gain` - a plain daily search
    # churned too much to beat once a week. Measured at 0.25 with the horizon
    # and no search on the look-ahead day (add gate `odds-daily-h-f25-x1`, 4
    # drafts x 2 seasons, against the weekly search alone): ahead in 6 of 8
    # leagues, clearly in 4, none clearly behind (worst -0.13 +/- 0.07);
    # pooled +0.12 +/- 0.02 categories a week, ~6 more adds a season. Not
    # positive in both seasons for any one draft, so the default waits on the
    # manager; the evidence says turn it on.
    mid_week_floor: float | None = None
    # A mid-week move must also pay over the next week, the two together
    # clearing `mid_week_floor` - a pickup that only lasts until Sunday costs
    # the roster spot a week later.
    mid_week_horizon: bool = True
    # Never propose dropping one of next season's best keepers: the top
    # n_keepers + keeper_margin eligible players by projected value next
    # season, less what keeping costs (`season.keeper_value`). Decided once a
    # week, so the line does not move under a waiting proposal. A league
    # without keepers protects nobody.
    #
    # Measured twice. Which keepers it picks (`ppilot season keeper-check`,
    # 48 drafted rosters, 2023-24 and 2024-25): from January on, those it would
    # keep did better the next season than the preseason projection's picks
    # and far better than the per-game value drops were judged by. What it
    # costs now (add gate `odds-daily-h-f25-x1-k1`, 12 teams x 22 weeks):
    # +0.04 +/- 0.05 and +0.14 +/- 0.05 categories a week - nothing; the
    # players it shields are the rising ones the per-game value cut too soon.
    protect_keepers: bool = True
    keeper_margin: int = 1
    # How the search shortlists free agents before pricing them properly.
    # "value": the 20 worth most per game times the empty slots they would
    # fill. "union": those, plus the 20 whose games move this week's close
    # categories most (`odds.leverage`) - a hitter when HIT is level, which a
    # general value can rank out of the 20.
    #
    # "value", on measurement (add gate `-e` against the live arm, 4 drafts x
    # 2 seasons): behind in 7 of 8 leagues, pooled -0.044 +/- 0.023
    # categories a week, with 25-70 more adds a league-season. A value-only
    # list of 40 (`-n40`, the control) did the same: a longer list finds more
    # moves that clear the floor on paper and do not pay.
    screen: str = "value"

    def __post_init__(self) -> None:
        if self.screen not in SCREENS:
            raise AuthorityError(
                f"authority.transactions.screen must be one of {', '.join(SCREENS)}; "
                f"got {self.screen!r}"
            )
        if self.add_scoring not in ADD_SCORING:
            raise AuthorityError(
                f"authority.transactions.add_scoring must be one of {', '.join(ADD_SCORING)}; "
                f"got {self.add_scoring!r}"
            )
        if self.playoff_reserve < 0:
            raise AuthorityError("authority.transactions.playoff_reserve cannot be negative")
        for name in ("preload_days", "look_ahead_days"):
            if not 0 <= getattr(self, name) <= 6:
                raise AuthorityError(f"authority.transactions.{name} must be between 0 and 6")
        if self.mid_week_floor is not None and self.mid_week_floor < 0:
            raise AuthorityError("authority.transactions.mid_week_floor cannot be negative")
        if self.keeper_margin < 0:
            raise AuthorityError("authority.transactions.keeper_margin cannot be negative")

    @property
    def requires_approval(self) -> bool:
        """Always true. A property rather than a field so it cannot be set."""
        return True

    def describe(self) -> list[str]:
        return [
            "Transactions (add / drop / claim): APPROVAL REQUIRED, always.",
            (
                f"  proposed only when it adds {self.min_expected_gain:.2f} expected categories"
                if self.add_scoring == "odds"
                else f"  proposed only when it closes {self.min_weekly_gain:.0%} of a live gap"
            ),
            f"  at most {self.max_pending} awaiting your decision at once",
            f"  {self.playoff_reserve} acquisition(s) held back for the playoffs",
            (
                f"  on the week's last {self.look_ahead_days} day(s), a waiting add must pay "
                f"over the rest of the week and the next one together"
                if self.look_ahead_days
                else "  a waiting add is judged on the current week alone"
            ),
            (
                f"  on the week's last {self.preload_days} day(s), the search also prices "
                f"against next week (measured worse - off by default)"
                if self.preload_days
                else "  the search prices against the current week"
            ),
            (
                f"  mid-week: a search every morning; a swap must add "
                f"{self.mid_week_floor:.2f} expected categories"
                + (" over the rest of this week and next together" if self.mid_week_horizon else "")
                if self.mid_week_floor is not None
                else "  mid-week: a search only when a roster spot opens"
            ),
            (
                f"  next season's best keepers, and {self.keeper_margin} more, are never "
                f"proposed as a drop"
                if self.protect_keepers
                else "  next season's keepers are not protected from a drop"
            ),
        ]


@dataclass(frozen=True)
class Authority:
    lineup: LineupAuthority = field(default_factory=LineupAuthority)
    transactions: TransactionAuthority = field(default_factory=TransactionAuthority)

    def describe(self) -> str:
        return "\n".join([*self.lineup.describe(), "", *self.transactions.describe()])

    @classmethod
    def from_config(cls, cfg: dict) -> Authority:
        """Parse an `[authority]` table. Unknown keys are refused, not ignored.

        A misspelled bound that silently keeps the default is how a tool ends up
        acting under criteria nobody agreed to.
        """
        lineup_cfg = dict(cfg.get("lineup", {}) or {})
        tx_cfg = dict(cfg.get("transactions", {}) or {})

        if "enabled" in tx_cfg:
            raise AuthorityError(
                "authority.transactions.enabled is not a setting: transactions always "
                "require an approved proposal. Remove it."
            )
        _refuse_removed("authority.lineup", lineup_cfg)
        _refuse_unknown("authority.lineup", lineup_cfg, LineupAuthority)
        _refuse_unknown("authority.transactions", tx_cfg, TransactionAuthority)

        if "never_bench" in lineup_cfg:
            lineup_cfg["never_bench"] = tuple(str(n) for n in lineup_cfg["never_bench"])
        return cls(
            lineup=LineupAuthority(**lineup_cfg),
            transactions=TransactionAuthority(**tx_cfg),
        )


# Settings that existed and should not be silently ignored if someone still
# has them in a config: say why they went rather than "unknown setting".
REMOVED = {
    "max_swaps_per_day": (
        "lineup changes are free and unlimited until each player's game starts, "
        "so capping them per day modelled a constraint that does not exist. "
        "The real bound is the lock time, which is reported instead."
    ),
}


def _refuse_removed(where: str, cfg: dict) -> None:
    for key, why in REMOVED.items():
        if key in cfg:
            raise AuthorityError(f"{where}.{key} has been removed: {why}")


def _refuse_unknown(where: str, cfg: dict, cls: type) -> None:
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise AuthorityError(
            f"{where}: unknown setting(s) {', '.join(unknown)}. Known: {', '.join(sorted(known))}"
        )
