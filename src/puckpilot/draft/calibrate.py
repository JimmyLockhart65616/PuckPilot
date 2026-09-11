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
    n_skipped: int = 0

    @property
    def text(self) -> str:
        if not self.losses:
            if self.n_skipped:
                return (
                    f"{self.n_skipped} harvested draft(s) carried no usable ADP, so none of "
                    "them says anything about survival.\nBuild the Yahoo player map first "
                    "(`ppilot yahoo playermap`) so pool ADP is available to fit against."
                )
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


def observations_from(
    result: MockResult,
    max_lookahead: int = MAX_LOOKAHEAD,
    adp: dict[str, float] | None = None,
) -> list[Observation]:
    """Turn one draft into survive/gone observations.

    `adp` is the pre-draft market rank per Yahoo player id, and it must come
    from OUTSIDE this draft. Pass Yahoo's published pool ADP
    (`pool_adp(conn, league_key)`) when it is available: the in-draft
    advice values the room broadcasts mid-draft cover only a handful of players
    - 26 of 192 in the 2026-09-08 mock - and are no longer recorded at all (see
    `farm.MockRecorder.ingest`).

    There is deliberately NO fallback to draft order. Setting a player's rank to
    the pick he went at makes survival a tautology (`survived` is then exactly
    `target <= rank`), and the log-loss of a tautology falls monotonically as the
    spread shrinks - on the real 2026-09-08 draft it collapsed to the bottom of
    the grid at a loss of 0.06, against 0.46 for the same draft fitted on real
    ADP. Pooled with honest drafts it would drag the whole fit toward zero. A
    draft with no external ADP says nothing about survival, so it contributes
    nothing.
    """
    picked_at: dict[str, int] = {}
    for row in result.picks:
        yahoo_id = str(row.get("yahoo_id"))
        pick = int(row.get("pick", 0))
        if yahoo_id and pick:
            picked_at.setdefault(yahoo_id, pick)
    if not picked_at:
        return []

    if adp is None:
        adp = {
            str(row["yahoo_id"]): float(row["adp_rank"])
            for row in result.adp_observations
            if row.get("adp_rank") is not None
        }
    if not adp:
        return []

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
    adp: dict[str, float] | None = None,
) -> CalibrationReport:
    """Fit survival_spread across every harvested draft.

    `adp` is Yahoo's published pool ADP (see `observations_from`). Without it
    only the sparse in-draft advice channel is available, which covers a small
    fraction of each draft.

    `sample_every` thins the observations: consecutive target picks for one
    player are near-duplicates, and keeping all of them would overstate the
    sample without adding information.
    """
    observations: list[Observation] = []
    used = 0
    skipped = 0
    for result in results:
        rows = observations_from(result, adp=adp)
        if not rows:
            # No external ADP for this draft. Counted and reported rather than
            # quietly dropped, so a harvest that contributed nothing is visible.
            skipped += 1
            continue
        used += 1
        observations.extend(rows[::sample_every])

    report = CalibrationReport(
        n_observations=len(observations),
        n_drafts=used,
        incumbent=incumbent,
        n_skipped=skipped,
    )
    if not observations:
        return report
    report.losses = {spread: _log_loss(observations, spread) for spread in grid}
    report.best = min(report.losses, key=lambda s: report.losses[s])
    return report
