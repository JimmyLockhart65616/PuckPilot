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


class AuthorityError(ValueError):
    """The authority config is not a thing the tool can act on."""


@dataclass(frozen=True)
class LineupAuthority:
    """Standing permission to change today's lineup, and its bounds."""

    enabled: bool = False
    # Leave a lineup alone unless the swap is worth this much, in the same
    # z-like per-day units `GameValueModel` produces. Churning for 0.01 costs
    # nothing but makes the audit log unreadable and the tool untrustworthy.
    min_gain: float = 0.15
    # Never start a goalie less likely than this to actually start. A goalie who
    # does not play scores zero in four of twelve categories.
    min_goalie_p_start: float = 0.5
    # "never" benches anyone carrying a status; "always" ignores the flag;
    # "only_if_needed" starts him when the alternative is an empty slot.
    start_questionable: str = "only_if_needed"
    # Yahoo's own per-week minimum (min_games_played) is a rule, not a
    # preference - falling short forfeits the category.
    enforce_min_games: bool = True
    # A day that wants more changes than this is a day something is wrong
    # (a bad feed, a stale roster). Report instead of acting.
    max_swaps_per_day: int = 4
    # Names, because a person thinks in names. Config, never code.
    never_bench: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start_questionable not in QUESTIONABLE_MODES:
            raise AuthorityError(
                f"authority.lineup.start_questionable must be one of "
                f"{', '.join(QUESTIONABLE_MODES)}; got {self.start_questionable!r}"
            )
        if not 0.0 <= self.min_goalie_p_start <= 1.0:
            raise AuthorityError("authority.lineup.min_goalie_p_start must be between 0 and 1")
        if self.max_swaps_per_day < 0:
            raise AuthorityError("authority.lineup.max_swaps_per_day must not be negative")

    def describe(self) -> list[str]:
        """The criteria in words, for printing before acting under them."""
        if not self.enabled:
            return ["Lineup changes: RECOMMEND ONLY (no standing authority granted)."]
        return [
            "Lineup changes: AUTONOMOUS within these criteria -",
            f"  swap only when it gains at least {self.min_gain:.2f} for the day",
            f"  never start a goalie below {self.min_goalie_p_start:.0%} to start",
            f"  players flagged day-to-day: {self.start_questionable.replace('_', ' ')}",
            f"  weekly goalie minimum enforced: {'yes' if self.enforce_min_games else 'no'}",
            f"  at most {self.max_swaps_per_day} changes a day, then ask instead",
            *([f"  never benched: {', '.join(self.never_bench)}"] if self.never_bench else []),
        ]


@dataclass(frozen=True)
class TransactionAuthority:
    """Bounds on what may be *proposed*. There is deliberately no `enabled`.

    Adds, drops and claims are never executed without an approved proposal row,
    so this class tunes what reaches a person, not what happens without one.
    """

    # Do not propose a move worth less than this for the week; a proposal the
    # person will reject is a notification that trains them to ignore the next.
    min_weekly_gain: float = 0.5
    # More pending proposals than this means the tool is guessing, not advising.
    max_pending: int = 5

    @property
    def requires_approval(self) -> bool:
        """Always true. A property rather than a field so it cannot be set."""
        return True

    def describe(self) -> list[str]:
        return [
            "Transactions (add / drop / claim): APPROVAL REQUIRED, always.",
            f"  proposed only when worth at least {self.min_weekly_gain:.2f} for the week",
            f"  at most {self.max_pending} awaiting your decision at once",
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
        _refuse_unknown("authority.lineup", lineup_cfg, LineupAuthority)
        _refuse_unknown("authority.transactions", tx_cfg, TransactionAuthority)

        if "never_bench" in lineup_cfg:
            lineup_cfg["never_bench"] = tuple(str(n) for n in lineup_cfg["never_bench"])
        return cls(
            lineup=LineupAuthority(**lineup_cfg),
            transactions=TransactionAuthority(**tx_cfg),
        )


def _refuse_unknown(where: str, cfg: dict, cls: type) -> None:
    known = set(cls.__dataclass_fields__)
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise AuthorityError(
            f"{where}: unknown setting(s) {', '.join(unknown)}. Known: {', '.join(sorted(known))}"
        )
