"""Fit the survival model to how humans actually draft.

`RosterValuePolicy.survival` asks: given a player at ADP rank R, will he still be
there at pick P? It answers with a logistic,

    P(survive) = 1 / (1 + exp(-(adp_rank - next_pick_no) / survival_spread))

and `survival_spread = 6.0` is a guess, tuned against a *simulated* ADP bot
field. STATUS.md credits `survival_discount` with the engine's whole measured
edge, so that constant carries real weight on a guess.

Harvested mock drafts answer the question directly. For every player and every
pick number, we know whether he was still on the board — that is a plain
observation of the room's behaviour, needs no season outcome, and is not
circular with our own valuation.

The fit is a coarse grid on log-loss rather than an optimizer: the parameter is
one-dimensional and bounded, the sample is a few thousand rows, and a grid is
inspectable in a way a solver's output is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from puckpilot.draft.farm import MockResult

# Candidate spreads. 6.0 is the incumbent, so it must appear in the grid or the
# comparison cannot be honest.
GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 26.0)
# Sampling every (player, pick) pair would swamp the fit with far-away picks
# whose answer is never in doubt. Only look ahead a realistic gap.
MAX_LOOKAHEAD = 30


@dataclass
class Observation:
    """Did a player at `adp_rank` survive until pick `target`?"""

    adp_rank: float
    target_pick: int
    survived: bool


@dataclass
class CalibrationReport:
    n_observations: int = 0
    n_drafts: int = 0
    losses: dict[float, float] = field(default_factory=dict)
    best: float | None = None
    incumbent: float = 6.0

    @property
    def text(self) -> str:
        if not self.losses:
            return (
                "No usable observations. Harvest mock drafts first (`ppilot draft farm --runs N`)."
            )
        lines = [
            f"Survival calibration: {self.n_observations} observations "
            f"from {self.n_drafts} draft(s)",
            "",
            f"{'spread':>8}{'log-loss':>12}",
            "-" * 20,
        ]
        for spread in sorted(self.losses):
            mark = ""
            if spread == self.best:
                mark = "  <- best fit"
            elif spread == self.incumbent:
                mark = "  (current)"
            lines.append(f"{spread:>8.1f}{self.losses[spread]:>12.4f}{mark}")

        incumbent_loss = self.losses.get(self.incumbent)
        lines.append("")
        if self.best is None or incumbent_loss is None:
            return "\n".join(lines)
        if self.best == self.incumbent:
            lines.append(f"Keep survival_spread = {self.incumbent}: the data agrees with it.")
        else:
            delta = incumbent_loss - self.losses[self.best]
            lines.append(
                f"Data prefers survival_spread = {self.best} "
                f"(log-loss {self.losses[self.best]:.4f} vs {incumbent_loss:.4f}, "
                f"{delta:+.4f})."
            )
            if delta < 0.005:
                lines.append(
                    "That margin is small; treat it as agreement, not a mandate to change."
                )
            if self.n_drafts < 5:
                lines.append(
                    f"Only {self.n_drafts} draft(s) harvested - collect more before "
                    "moving a tuned constant."
                )
        return "\n".join(lines)


def observations_from(result: MockResult, max_lookahead: int = MAX_LOOKAHEAD) -> list[Observation]:
    """Turn one draft into survive/gone observations.

    `adp_rank` here is the player's position in *this* draft's own consensus
    order, approximated by Yahoo's published ADP when present and otherwise by
    the order players actually went. The question the model asks is about the
    room, so the room's own ordering is the right yardstick.
    """
    picked_at: dict[str, int] = {}
    for row in result.picks:
        yahoo_id = str(row.get("yahoo_id"))
        pick = int(row.get("pick", 0))
        if yahoo_id and pick:
            picked_at.setdefault(yahoo_id, pick)
    if not picked_at:
        return []

    adp = {
        str(row["yahoo_id"]): float(row["adp_rank"])
        for row in result.adp_observations
        if row.get("adp_rank") is not None
    }
    if not adp:
        # Fall back to draft order: with no published ADP, when a player went is
        # the only statement this draft makes about where the market valued him.
        adp = {pid: float(pick) for pid, pick in picked_at.items()}

    last_pick = max(picked_at.values())
    out: list[Observation] = []
    for yahoo_id, gone_at in picked_at.items():
        rank = adp.get(yahoo_id)
        if rank is None:
            continue
        # Window centred on the player's ADP, not on when he happened to go.
        # Anchoring on the outcome and capping only the forward side
        # over-samples "he lasted past his ADP" relative to reaches, which
        # biases the fit toward a flatter curve - measured at +25-30% on
        # synthetic drafts with a known spread.
        lo = max(1, int(rank) - max_lookahead)
        hi = min(last_pick, int(rank) + max_lookahead)
        for target in range(lo, hi + 1):
            out.append(Observation(rank, target, survived=target <= gone_at))
    return out


def _log_loss(observations: list[Observation], spread: float) -> float:
    """Mean negative log-likelihood of the logistic at this spread."""
    total, eps = 0.0, 1e-9
    for obs in observations:
        x = (obs.adp_rank - obs.target_pick) / spread
        # clamp to avoid overflow on far-from-the-boundary observations
        p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))
        p = min(1.0 - eps, max(eps, p))
        total += -math.log(p if obs.survived else 1.0 - p)
    return total / len(observations) if observations else float("inf")


def calibrate(
    results: list[MockResult],
    grid: tuple[float, ...] = GRID,
    incumbent: float = 6.0,
    sample_every: int = 7,
) -> CalibrationReport:
    """Fit survival_spread across every harvested draft.

    `sample_every` thins the observations: consecutive target picks for one
    player are near-duplicates, and keeping all of them would overstate the
    sample without adding information.
    """
    observations: list[Observation] = []
    used = 0
    for result in results:
        rows = observations_from(result)
        if not rows:
            continue
        used += 1
        observations.extend(rows[::sample_every])

    report = CalibrationReport(n_observations=len(observations), n_drafts=used, incumbent=incumbent)
    if not observations:
        return report
    report.losses = {spread: _log_loss(observations, spread) for spread in grid}
    report.best = min(report.losses, key=lambda s: report.losses[s])
    return report
