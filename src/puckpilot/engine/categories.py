from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str  # column name in aggregate/projection frames
    label: str  # Yahoo-style display label
    kind: str  # 'skater' | 'goalie'
    higher_is_better: bool = True
    rate: bool = False  # rate stats are volume-weighted in valuation, never summed


# Yahoo default H2H categories. The real league's set replaces these once
# Yahoo API access lands (league settings -> Category list).
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
