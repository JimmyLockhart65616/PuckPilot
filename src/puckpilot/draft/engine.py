from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from puckpilot.engine.valuation import DEFAULT_SHAPE, LeagueShape


@dataclass(frozen=True)
class DraftRules:
    """Roster construction rules for a snake draft.

    caps stop bots from hoarding a position; mins guarantee a startable roster
    (picks are forced to needy positions once picks_left equals unmet minimums).
    """

    shape: LeagueShape = DEFAULT_SHAPE
    rounds: int = 18  # 12 skater starters + 2 G + 4 bench
    caps: dict[str, int] = field(default_factory=lambda: {"C": 4, "L": 4, "R": 4, "D": 6, "G": 3})
    mins: dict[str, int] = field(default_factory=lambda: {"C": 2, "L": 2, "R": 2, "D": 4, "G": 2})


def eligible_positions(counts: dict[str, int], rules: DraftRules, picks_left: int) -> set[str]:
    """Positions a team may draft now: under cap, and forced to unmet minimums
    when there are only just enough picks left to satisfy them."""
    allowed = {p for p, cap in rules.caps.items() if counts.get(p, 0) < cap}
    needed = {p: max(0, m - counts.get(p, 0)) for p, m in rules.mins.items()}
    if sum(needed.values()) >= picks_left:
        allowed &= {p for p, n in needed.items() if n > 0}
    return allowed


class Universe:
    """Array-backed draft pool for fast simulated picks.

    Built from a rank_players() frame joined with an adp_rank column.
    Row order is the frame's order (vorp descending).
    """

    def __init__(self, ranked: pd.DataFrame):
        self.frame = ranked  # full source columns (proj_gp, per-cat totals, ...)
        self.ids = ranked.index.to_numpy()
        self.names = ranked["name"].to_numpy()
        self.pos = ranked["position"].to_numpy()
        self.vorp = ranked["vorp"].to_numpy(dtype=float)
        self.z_total = ranked["z_total"].to_numpy(dtype=float)
        self.adp_rank = (
            ranked["adp_rank"].to_numpy(dtype=float)
            if "adp_rank" in ranked
            else np.full(len(ranked), 999.0)
        )
        z_cols = [c for c in ranked.columns if c.startswith("z_") and c != "z_total"]
        self.z_by_cat = {
            c.removeprefix("z_"): ranked[c].fillna(0.0).to_numpy(dtype=float) for c in z_cols
        }

    def __len__(self) -> int:
        return len(self.ids)


def _pick_best(
    u: Universe,
    avail: np.ndarray,
    counts: dict[str, int],
    rules: DraftRules,
    picks_left: int,
    score: np.ndarray,
) -> int:
    """Index of the highest-score available player at an eligible position."""
    allowed = eligible_positions(counts, rules, picks_left)
    mask = avail & np.isin(u.pos, list(allowed))
    if not mask.any():
        mask = avail
    masked = np.where(mask, score, -np.inf)
    return int(np.argmax(masked))


class VorpPolicy:
    """Best available by VORP within roster rules (baseline; roster-blind)."""

    name = "vorp"

    def pick(self, u, avail, counts, rules, picks_left, rng, ctx=None) -> int:
        return _pick_best(u, avail, counts, rules, picks_left, u.vorp)


class RosterValuePolicy:
    """The engine: marginal value vs the current roster build + market timing.

    A candidate scores full z while a starting slot (position or util) is open
    for them, and only bench_factor of it once they'd ride the pine. On top,
    survival_discount devalues players the ADP-following room will likely leave
    for our next turn (logistic in adp_rank around our next pick number).

    Defaults are what the 2025-26 walk-forward draft sim selected empirically:
    goalie_weight 1.0 (both boosting goalies for category leverage and fading
    them for projection noise LOWERED finish rates), no cat reliability weights
    (conceding unpredictable cats costs more than refocusing gains), and
    survival_discount 0.3 (the one knob that beat the bot field, p<0.005).
    """

    name = "engine"

    def __init__(
        self,
        goalie_weight: float = 1.0,
        bench_factor: float = 0.15,
        cat_weights: dict[str, float] | None = None,
        survival_discount: float = 0.3,
        survival_spread: float = 6.0,
    ):
        self.goalie_weight = goalie_weight
        self.bench_factor = bench_factor
        self.cat_weights = cat_weights
        self.survival_discount = survival_discount
        self.survival_spread = survival_spread

    def _base_score(self, u: Universe) -> np.ndarray:
        if not self.cat_weights:
            return u.z_total.copy()
        score = np.zeros(len(u))
        for cat, z in u.z_by_cat.items():
            score += self.cat_weights.get(cat, 1.0) * z
        return score

    def pick(self, u, avail, counts, rules, picks_left, rng, ctx=None) -> int:
        slots = dict(rules.shape.slots)
        util_used = sum(max(0, counts.get(p, 0) - s) for p, s in slots.items() if p != "G")
        util_open = util_used < rules.shape.util_slots

        score = self._base_score(u)
        goalie_mask = u.pos == "G"
        score[goalie_mask] *= self.goalie_weight
        for pos, slot_count in slots.items():
            if counts.get(pos, 0) < slot_count:
                continue  # starting slot open -> full value
            if pos != "G" and util_open:
                continue  # overflows into an open util slot
            benched = u.pos == pos
            # scale positive value only: a bad player doesn't get better by sitting
            score[benched] = np.where(
                score[benched] > 0, score[benched] * self.bench_factor, score[benched]
            )

        if self.survival_discount and ctx and ctx.get("next_pick_no") is not None:
            # discount players the ADP-following room will likely leave for our
            # next turn — spend this pick where the market is about to strike
            taken_by_next = ctx["next_pick_no"]
            p_survive = 1.0 / (1.0 + np.exp(-(u.adp_rank - taken_by_next) / self.survival_spread))
            factor = 1.0 - self.survival_discount * p_survive
            score = np.where(score > 0, score * factor, score)
        return _pick_best(u, avail, counts, rules, picks_left, score)


class AdpBot:
    """Drafts by (pseudo-)ADP with per-pick gaussian noise on the rank."""

    name = "adp"

    def __init__(self, noise_sd: float = 4.0):
        self.noise_sd = noise_sd

    def pick(self, u, avail, counts, rules, picks_left, rng, ctx=None) -> int:
        noisy = -(u.adp_rank + rng.normal(0.0, self.noise_sd, len(u)))
        return _pick_best(u, avail, counts, rules, picks_left, noisy)


class GreedyZBot:
    """Best available by raw z_total — ignores replacement level."""

    name = "greedy_z"

    def pick(self, u, avail, counts, rules, picks_left, rng, ctx=None) -> int:
        return _pick_best(u, avail, counts, rules, picks_left, u.z_total)


class PuntBot:
    """Ignores one or more categories and maximizes z over the rest."""

    name = "punt"

    def __init__(self, punt: tuple[str, ...] = ("plus_minus",)):
        self.punt = punt

    def pick(self, u, avail, counts, rules, picks_left, rng, ctx=None) -> int:
        score = u.z_total.copy()
        for cat in self.punt:
            if cat in u.z_by_cat:
                score = score - u.z_by_cat[cat]
        return _pick_best(u, avail, counts, rules, picks_left, score)
