from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str  # column name in aggregate/projection frames
    label: str  # Yahoo-style display label
    kind: str  # 'skater' | 'goalie'
    higher_is_better: bool = True
    rate: bool = False  # rate stats are volume-weighted in valuation, never summed


# Every category the engines know how to score, keyed by the label leagues use
# in their config files. Add an entry here (plus its stat mapping in
# aggregate.SKATER_LOG_KEYS / replay.LOG_KEYS) to support a new category.
CATALOG: dict[str, Category] = {
    c.label: c
    for c in (
        Category("goals", "G", "skater"),
        Category("assists", "A", "skater"),
        Category("points", "P", "skater"),
        Category("plus_minus", "+/-", "skater"),
        Category("pim", "PIM", "skater"),
        Category("ppp", "PPP", "skater"),
        Category("shp", "SHP", "skater"),
        Category("gwg", "GWG", "skater"),
        Category("sog", "SOG", "skater"),
        Category("hits", "HIT", "skater"),
        Category("blocks", "BLK", "skater"),
        Category("wins", "W", "goalie"),
        Category("saves", "SV", "goalie"),
        # Yahoo scores SA as higher-is-better (stat_id 24, sort_order=1): as a
        # counting stat it rewards workload, not weak goaltending.
        Category("shots_against", "SA", "goalie"),
        Category("shutouts", "SHO", "goalie"),
        Category("save_pct", "SV%", "goalie", rate=True),
        Category("gaa", "GAA", "goalie", higher_is_better=False, rate=True),
    )
}


class UnknownCategory(KeyError):
    pass


def resolve(label: str) -> Category:
    """Look up a category by its config label (e.g. 'SV%'), case-insensitively."""
    for key in (label, label.upper()):
        if key in CATALOG:
            return CATALOG[key]
    raise UnknownCategory(
        f"unknown category {label!r}; known categories: {', '.join(sorted(CATALOG))}"
    )


# Yahoo default H2H categories, used when a league config omits them.
SKATER_CATS_DEFAULT = (
    Category("goals", "G", "skater"),
    Category("assists", "A", "skater"),
    Category("plus_minus", "+/-", "skater"),
    Category("pim", "PIM", "skater"),
    Category("ppp", "PPP", "skater"),
    Category("sog", "SOG", "skater"),
)

GOALIE_CATS_DEFAULT = (
    Category("wins", "W", "goalie"),
    Category("gaa", "GAA", "goalie", higher_is_better=False, rate=True),
    Category("save_pct", "SV%", "goalie", rate=True),
    Category("shutouts", "SHO", "goalie"),
)
