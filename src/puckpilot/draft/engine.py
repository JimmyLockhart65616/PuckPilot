from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from puckpilot.engine.valuation import DEFAULT_SHAPE, LeagueShape, replacement_level


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

    def with_adp(self, adp_rank: np.ndarray) -> Universe:
        """Shallow view sharing all arrays but carrying a different ADP.

        Used to re-base ADP after keepers come off the board: the shared arrays
        make this cheap enough to call once per simulated draft.
        """
        u = copy.copy(self)
        u.adp_rank = adp_rank
        return u


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

    Defaults come from the 2025-26 walk-forward draft sim under the real league
    (H2H categories, 12 teams, 3 keepers). BOTH tunable knobs reversed direction
    when the objective moved from roto to H2H:

    - bench_factor 0.70 (was 0.15). With 3 bench spots on a 16-man roster almost
      every drafted player accumulates stats, so heavily discounting "bench"
      value was calibrated for a roster shape this league does not have.
    - survival_discount 0.50 (was 0.30). H2H rewards winning categories outright,
      so spending picks where the market is about to strike matters more.

    Confirmed on a held-out seed: top-3 rate 0.342 vs the best bot's 0.272
    (n=600, p=0.0002). The old roto defaults FAIL the same test (0.265, p=0.70).

    Re-tested under H2H and still rejected, as under roto: goalie_weight in
    either direction (0.85 -> 0.310, 1.15 -> 0.333 vs 0.350 baseline), and
    category weights — damping the collinear P=G+A costs value (0.280) and
    chasing HIT/BLK is disastrous (0.130), because the peripheral categories are
    cheap to acquire on waivers but scoring is not.

    Both knobs were re-screened on the VORP basis, because they correct the
    shape of the base score and the base score changed (see `_base_score`). A
    5x5 grid found a clean interior peak:

    - survival_discount 0.50 -> **0.30**. Monotone either side (0.20 and 0.40
      both lower, 0.0 clearly worse, so market timing still earns its keep - at
      a smaller weight). Confirmed at n=1000 on two seeds not used for the grid:
      top-3 0.530/0.536 -> 0.640/0.665 on target 2025-26 with non-overlapping
      CIs, and 0.387/0.400 -> 0.395/0.414 on 2024-25. Same sign in all four
      runs; most of the magnitude is 2025-26-specific, and that is stated rather
      than averaged away.
    - bench_factor stays 0.70. The grid is flat across 0.40-0.70 and 0.70 sits
      inside that plateau; 0.55 edged it on both confirmation seeds but well
      inside overlapping CIs, which is not evidence to move a tuned constant.
    - goalie_weight 1.0 -> **1.5**, and this one is structural rather than
      cosmetic. VORP subtracts a positional replacement level, and with only 24
      starting goalie slots drawn from a deep pool the replacement goalie is
      already good, so goalie VORP compresses hard. But the goalie categories
      are four of twelve - a third of every matchup - carried by two roster
      spots. Positional scarcity and category weight are different quantities,
      and VORP only knows the first.

      Left uncorrected the engine conceded all four: the per-category diagnostic
      showed SV .613 -> .289, SA .614 -> .288, W .514 -> .294 when the basis
      moved. At 1.5 the worst category recovers from .274 to .450 and the engine
      wins 8-9 of 12 rather than 5-6.

      1.5 is not the argmax. 2.0-2.5 score a higher top-3 on 2025-26 but win
      only five categories, trading a balanced roster for big margins in a few -
      a worse bet against eleven humans whose tendencies we do not know. 1.5 has
      the best or equal mean finish on both target seasons and all four
      confirmation seeds. Out-of-sample top-3 on 2024-25 is a wash between all
      three values; the magnitude of the gain is 2025-26-specific even though
      the defect it repairs is not.
    """

    name = "engine"

    def __init__(
        self,
        goalie_weight: float = 1.5,
        bench_factor: float = 0.70,
        cat_weights: dict[str, float] | None = None,
        survival_discount: float = 0.30,
        survival_spread: float = 6.0,
        basis: str = "vorp",
        replacement_depth: float = 0.0,
    ):
        if basis not in ("vorp", "z"):
            raise ValueError(f"basis must be 'vorp' or 'z', got {basis!r}")
        if replacement_depth < 0:
            raise ValueError(f"replacement_depth must be >= 0, got {replacement_depth!r}")
        self.goalie_weight = goalie_weight
        self.bench_factor = bench_factor
        self.cat_weights = cat_weights
        self.survival_discount = survival_discount
        self.survival_spread = survival_spread
        self.basis = basis
        self.replacement_depth = replacement_depth

    def _base_score(self, u: Universe, rules=None, avail=None) -> np.ndarray:
        """What a player is worth before roster shape and market timing.

        `basis="vorp"` subtracts a positional replacement level; `basis="z"` is
        raw summed z-score. The distinction is the whole point of VORP and this
        policy ignored it for a long time - `_base_score` returned `z_total`, so
        a defenceman and a centre with equal z looked identical even though the
        replacement-level defenceman is far worse. VORP was computed, displayed
        on the board, and then left out of the decision.

        Measured at n=1500 on a locked seed never used for tuning, against the
        real 12-category league, with non-overlapping CIs in both directions:

            target 2025-26   z 0.443 [.418,.468]   vorp 0.525 [.500,.551]
            target 2024-25   z 0.294 [.271,.318]   vorp 0.379 [.355,.404]

        Two target seasons, same sign, same rough magnitude - which is why this
        is the default and the goals-weight result that turned up alongside it
        is not: that one gained 0.29 on 2025-26 and *lost* on 2024-25, i.e. it
        was fitted to one season.

        `cat_weights` still applies on the z basis, where per-category weighting
        is meaningful. VORP is already a scalar, so the two do not compose.
        """
        if self.basis == "vorp" and not self.cat_weights:
            if avail is None or not self.replacement_depth or rules is None:
                return u.vorp.copy()
            return self._dynamic_vorp(u, rules, avail)
        if not self.cat_weights:
            return u.z_total.copy()
        score = np.zeros(len(u))
        for cat, z in u.z_by_cat.items():
            score += self.cat_weights.get(cat, 1.0) * z
        return score

    def survival(self, u: Universe, ctx=None) -> np.ndarray:
        """P(each player is still there at our next turn), per the ADP model.

        Exposed because the live console shows it: "he'll last, take the other
        guy" is the single most useful thing to put in front of a human on the
        clock. Returns all-zeros when there is no next pick (last round), which
        makes the discount vanish — nothing survives a draft that is over.
        """
        if not (ctx and ctx.get("next_pick_no") is not None):
            return np.zeros(len(u))
        taken_by_next = ctx["next_pick_no"]
        return 1.0 / (1.0 + np.exp(-(u.adp_rank - taken_by_next) / self.survival_spread))

    def score(self, u: Universe, counts, rules, ctx=None) -> np.ndarray:
        """Per-player score before availability and position eligibility.

        Split out of pick() so the live console can rank the whole board rather
        than only learn the argmax. pick() calls this, so the two can never drift.
        """
        slots = dict(rules.shape.slots)
        util_used = sum(max(0, counts.get(p, 0) - s) for p, s in slots.items() if p != "G")
        util_open = util_used < rules.shape.util_slots

        score = self._base_score(u, rules, ctx.get("avail") if ctx else None)
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
            factor = 1.0 - self.survival_discount * self.survival(u, ctx)
            score = np.where(score > 0, score * factor, score)
        return score

    def _dynamic_vorp(self, u: Universe, rules: DraftRules, avail: np.ndarray) -> np.ndarray:
        """VORP re-based against the players who are actually still on the board.

        The frozen `u.vorp` prices every player against the pool as it looked
        before the draft started. After a run on a position that is simply
        wrong: with eight defencemen left, the ninth-best D is not worth what
        the pre-draft table said.

        **This is OFF by default (`replacement_depth=0.0`) and the measurements
        below are why.** It is kept because the mechanism is real and clearly
        helps under some conditions, but it did not earn the default.

        Depth is a FIXED fraction of the league's starting slots at the
        position, not a countdown of how many are still to be taken. That
        distinction is the whole design, and the obvious alternative was
        measured and is worse in both directions:

            replacement depth            2025-26         2024-25
            static (frozen)              0.743 / 0.702   0.440 / 0.383
            fixed fraction (this)        0.713 / 0.727   0.510 / 0.498
            `slots - already_drafted`    0.568 / 0.537   0.347 / 0.337

        Tracking the D replacement level as the position empties shows why. A
        fixed depth falls monotonically (-3.56 at 0 drafted to -8.79 at 80), so
        the defencemen still on the board correctly gain value as the position
        thins. `slots - already_drafted` is flat at -6.10 from 24 through 48 -
        the meat of the draft - because the depth shrinks as fast as the pool
        does, leaving no gradient at all. It also asks the wrong question: "how
        many are left to be taken" is about timing, which `survival_discount`
        already handles, and the two fight.

        Why it is off. The pre-registered rule was: ship only if the sign is
        consistent across both target seasons and both selection seeds. It is
        not. n=800, depth 0.0 vs 0.5:

            target 2025-26   0.746 / 0.704   ->   0.642 / 0.656    worse
            target 2024-25   0.419 / 0.384   ->   0.557 / 0.566    much better

        Both seeds agree within each season, so this is not noise - the two
        seasons genuinely disagree. The obvious explanation was tested and
        refuted: 2024-25 has no real keeper list and falls back to simulated
        keepers, but re-running 2025-26 with simulated keepers still prefers
        static (0.755 / 0.710 vs 0.709 / 0.672), so it is not the keeper
        structure. No further explanation was found, and a mechanism that helps
        by +0.14 in one season and hurts by -0.09 in another for reasons nobody
        can name is not one to hand a live draft.

        What the drafter gets instead is the same information as a fact rather
        than a weight: the board shows how many startable players remain at each
        position and where the cliff is, and a human applies it. Set
        `replacement_depth=0.5` to turn the scoring effect on.
        """
        base = u.z_total.copy()
        starters = rules.shape.starters_by_pos()
        for pos, slots_at_pos in starters.items():
            at_pos = u.pos == pos
            here = avail & at_pos
            n = int(here.sum())
            if not n:
                continue
            depth = int(round(slots_at_pos * self.replacement_depth))
            if depth <= 0:
                continue
            vals = np.sort(u.z_total[here])[::-1]
            base[at_pos] -= replacement_level(vals, depth)
        return base

    def pick(self, u, avail, counts, rules, picks_left, rng, ctx=None) -> int:
        return _pick_best(u, avail, counts, rules, picks_left, self.score(u, counts, rules, ctx))


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
