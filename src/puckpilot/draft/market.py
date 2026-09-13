"""Market-implied value, for players we cannot project at all.

Every projection in this codebase needs game-log history to blend. A true
rookie has none, so `engine.projections.project` correctly leaves him out
entirely — not a bug, just a boundary. Left there, three things go wrong on
draft night: the room can draft a player our board has never heard of (the
pick clock drifts — see `draft.board.UnknownPlayerError`); the disagreement
panel and every "sleeper" list are blind to exactly the players sleeper-hunting
is about; and there is no honest way to say "the market prices him here" when
we have no opinion of our own.

The fix is not a projection. It is reading off what the market already knows:
Yahoo's own ADP, refined by real picks from harvested mock drafts where we have
them (`mock_consensus`), turned into an implied VORP via a curve fit against
players we DO have both a real VORP and a real market price for
(`fit_market_curve`). That curve is fit once per league/season and applied only
to the gap — it never touches `value_players` or `replacement_adjust`, so it
cannot move a single real player's number.

Three rules keep this honest, and all three are enforced by callers, not by
this module alone:

- **Never computed before real ranking, only appended after.** This module
  takes a fitted `Universe`'s own arrays as input; it does not re-rank anyone.
- **Rendered as a market price, not an opinion.** Every row this module builds
  carries `source="market"` — see `Candidate.source` and how the web view and
  `explain.py` treat it differently everywhere.
- **Excluded from "where do we disagree with the room."** If a player's own
  value IS the market's price, comparing our rank to the market's rank about
  him measures nothing — see `advice.market_disagreement`.
"""

from __future__ import annotations

import glob
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# Yahoo lists primary position first, then secondary eligibility, then Util/IR
# flags that are not positions at all — take the first token that is one.
_YAHOO_POS = {"C": "C", "LW": "L", "RW": "R", "D": "D", "G": "G"}


def primary_position(positions: str | None) -> str | None:
    """ "C,LW,RW,Util" -> "C". None if nothing recognizable is in the string.

    Only needed as a fallback for a market-priced player with no NHL id at all
    (so no `nhl_players.position` to trust instead) — see `build_market_frame`.
    """
    for tok in (positions or "").split(","):
        tok = tok.strip()
        if tok in _YAHOO_POS:
            return _YAHOO_POS[tok]
    return None


def _bare(player_key: str) -> str:
    """ "477.p.6743" -> "6743", the form the draft-room websocket and the
    harvested mock files both use."""
    return str(player_key).rsplit(".", 1)[-1]


def mock_consensus(mock_glob: str = "data/mocks/*.json") -> dict[str, list[float]]:
    """bare Yahoo id -> pick numbers seen across every harvested mock on disk,
    each normalized to a 12-team board (`pick * 12 / n_teams`) so a 14-team
    mock's picks are comparable to a 12-team one's.

    Missing or unparsable files are skipped rather than raised — this is a
    refinement on top of Yahoo ADP, never the only signal, so losing it must
    degrade gracefully to ADP alone rather than take the board down.
    """
    out: dict[str, list[float]] = {}
    for path in sorted(glob.glob(mock_glob)):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        n_teams = data.get("n_teams") or 12
        for p in data.get("picks", []):
            pick_no, yahoo_id = p.get("pick"), p.get("yahoo_id")
            if pick_no is None or yahoo_id is None:
                continue
            out.setdefault(str(yahoo_id), []).append(float(pick_no) * 12.0 / n_teams)
    return out


# Once a player has been drafted in this many independent mock rooms, the mock
# mean is trusted as much as Yahoo's own ADP. Below that, ADP dominates the
# blend in proportion to how thin the mock evidence still is. Chosen, not
# fitted: it is a statement about how much a single mock room's idiosyncrasy
# should be allowed to move a number, not a knob tuned against an outcome.
FULL_WEIGHT_OBSERVATIONS = 8


def consensus_rank(yahoo_adp: float, mock_picks: list[float]) -> float:
    """Blend Yahoo ADP with harvested-mock consensus, weighted by evidence.

    Checked against 12 completed harvested mocks before adopting this: Yahoo
    ADP and mock draft position agree in direction often enough (McKenna,
    Stenberg, Martone all seen in 12/12 mocks with tight clustering) that
    ignoring real human picks where we have them would be throwing away signal
    — but a single mock room is not a consensus (Hagens appeared in only 1 of
    12), so a lone data point must not dominate a broad market's ADP.

    Not simply "average every mock pick number": a player undrafted in some
    room's specific run and drafted in another is not evidence he goes right
    at the edge of that room's draft — the observations we DO have (where
    present) cluster tightly, so the mean of those is trusted, weighted by how
    many independent rooms produced it.
    """
    if not mock_picks:
        return float(yahoo_adp)
    weight = min(len(mock_picks), FULL_WEIGHT_OBSERVATIONS) / FULL_WEIGHT_OBSERVATIONS
    mock_mean = sum(mock_picks) / len(mock_picks)
    return (1.0 - weight) * float(yahoo_adp) + weight * mock_mean


# Below this many market-priced non-goalie rows, or fewer than two positions
# with enough of them, a per-position intercept is fitting noise, not a curve.
MIN_CURVE_ROWS = 20
MIN_ROWS_PER_POSITION = 5


@dataclass(frozen=True)
class MarketCurve:
    """log(consensus rank) -> implied VORP, one shared slope, one intercept per
    position. Fit once against the market-priced population that already has a
    real VORP, then applied only to players who have none.

    A shared slope rather than one per position is deliberate: the *marginal*
    value of ten ADP places is set by how bunched the pool is at that price,
    which is a property of the market as a whole, not of one position. What
    genuinely differs by position is the LEVEL — replacement level for a
    defenceman sits far below a centre's, so the same ADP means very different
    VORP depending on position — and that is exactly what the per-position
    intercept carries.
    """

    slope: float
    intercept: dict[str, float]
    r_squared: float
    n: int

    def predict(self, position: str, rank: float) -> float | None:
        b0 = self.intercept.get(position)
        if b0 is None or not (rank > 0):
            return None
        return b0 + self.slope * np.log(rank)


def fit_market_curve(
    vorp: np.ndarray,
    adp_rank: np.ndarray,
    position: np.ndarray,
    has_market: np.ndarray,
    exclude_positions: frozenset[str] = frozenset({"G"}),
) -> MarketCurve | None:
    """Fit `MarketCurve` against real (vorp, adp_rank, position) triples.

    Goalies excluded by default: goalie value depends on a team-strength blend
    (`GOALIE_TEAM_WIN_BLEND`) that has nothing to do with draft position, so the
    ADP/VORP relationship for goalies is a different, noisier shape and the
    unprojectable population this module targets is skaters-only in practice
    (the unmapped players beyond skaters are backup goalies nobody starts).

    Returns None rather than a bad fit when the population is too thin to
    support even one intercept per position reliably — a smaller or
    differently-shaped league must fall back cleanly, not extrapolate from a
    handful of points. That is a property of `LeagueConfig`, so it must not
    silently mis-value a league this was never checked against.
    """
    mask = has_market & (adp_rank > 0) & ~np.isin(position, list(exclude_positions))
    if int(mask.sum()) < MIN_CURVE_ROWS:
        return None

    x = np.log(adp_rank[mask])
    y = vorp[mask]
    pos = position[mask]

    counts = pd.Series(pos).value_counts()
    usable = sorted(counts[counts >= MIN_ROWS_PER_POSITION].index)
    if len(usable) < 2:
        return None
    keep = np.isin(pos, usable)
    x, y, pos = x[keep], y[keep], pos[keep]

    design = np.column_stack([*((pos == p).astype(float) for p in usable), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return MarketCurve(
        slope=float(beta[-1]),
        intercept=dict(zip(usable, (float(b) for b in beta[:-1]), strict=True)),
        r_squared=r_squared,
        n=int(len(y)),
    )


def build_market_frame(
    conn: sqlite3.Connection,
    league_key: str,
    universe,
    mock_glob: str = "data/mocks/*.json",
    progress: Progress = _noop,
) -> pd.DataFrame:
    """Synthetic rows for players the market prices but our own board cannot.

    `universe` must already carry real Yahoo ADP and `has_market` (i.e. it is
    the result of `build_live_board`'s `with_adp` + `has_market` assignment,
    not a bare `build_universe` call) — the curve is fit against ITS arrays, so
    whatever ADP the board is using is what the curve is fit against too.

    Returns a DataFrame shaped like a `Universe` source frame (name, position,
    team, vorp, z_total, adp_rank, age, train_gp, source, kind), indexed by NHL
    player id, ready to `pd.concat` onto the board's frame and rebuild a
    `Universe` from. Empty if the curve cannot be fit or nobody qualifies —
    always a valid, silently-do-nothing input to that concat.
    """
    curve = fit_market_curve(universe.vorp, universe.adp_rank, universe.pos, universe.has_market)
    if curve is None:
        progress("  market curve: too few market-priced players to fit, skipping")
        return pd.DataFrame()

    known_ids = {int(i) for i in universe.ids}
    rows = conn.execute(
        "SELECT player_key, full_name, team_abbrev, positions, nhl_player_id, adp_rank"
        " FROM yahoo_player_map WHERE league_key = ? AND adp_rank IS NOT NULL",
        (league_key,),
    ).fetchall()

    nhl_positions = dict(conn.execute("SELECT player_id, position FROM nhl_players"))
    ages = _bio_ages(conn)
    mocks = mock_consensus(mock_glob)

    out: dict[int, dict] = {}
    skipped_goalie = skipped_nopos = skipped_uncurved = 0
    next_synthetic = -1
    for r in rows:
        nhl_id = r["nhl_player_id"]
        if nhl_id is not None and int(nhl_id) in known_ids:
            continue  # already projectable — not this module's concern

        position = nhl_positions.get(int(nhl_id)) if nhl_id is not None else None
        if position is None:
            position = primary_position(r["positions"])
        if position == "G":
            skipped_goalie += 1
            continue
        if position is None:
            skipped_nopos += 1
            continue

        rank = consensus_rank(r["adp_rank"], mocks.get(_bare(r["player_key"]), []))
        vorp = curve.predict(position, rank)
        if vorp is None:
            skipped_uncurved += 1
            continue

        if nhl_id is not None:
            row_id = int(nhl_id)
        else:
            # No NHL id at all (should be rare after `sync_current_rosters`'
            # roster upsert, but not impossible for a genuinely undrafted
            # prospect) — a stable negative id, never colliding with a real
            # NHL one, so the row still has something to be indexed by.
            row_id = next_synthetic
            next_synthetic -= 1

        out[row_id] = {
            "name": r["full_name"],
            "position": position,
            "team": r["team_abbrev"] or "?",
            "adp_rank": rank,
            "vorp": vorp,
            "age": ages.get(nhl_id) if nhl_id is not None else None,
            "train_gp": 0.0,  # zero NHL evidence — flags "thin history" downstream
            "source": "market",
            "kind": "skater",
        }

    if skipped_goalie or skipped_nopos or skipped_uncurved:
        progress(
            f"  market-implied: skipped {skipped_goalie} goalie(s), "
            f"{skipped_nopos} with no resolvable position, "
            f"{skipped_uncurved} outside the curve's positions"
        )
    if out:
        progress(
            f"  market-implied: {len(out)} player(s) priced from the market "
            f"(curve R^2={curve.r_squared:.2f}, n={curve.n})"
        )
    frame = pd.DataFrame.from_dict(out, orient="index")
    frame.index.name = "player_id"
    if not frame.empty:
        # z_total is not used by default scoring (`basis="vorp"` reads `vorp`
        # directly) but is carried on every real row, so back-fill it rather
        # than leave a market row the one with a hole in it: vorp = z_total -
        # replacement_level(position), a fixed offset per position already
        # implied by the real, non-market rows.
        real = universe.has_market & ~np.isin(universe.pos, ["G"])
        offset = {
            p: float(np.mean((universe.z_total - universe.vorp)[real & (universe.pos == p)]))
            for p in frame["position"].unique()
            if (real & (universe.pos == p)).any()
        }
        frame["z_total"] = frame["vorp"] + frame["position"].map(offset).fillna(0.0)
    return frame


def _bio_ages(conn: sqlite3.Connection, reference_year: int | None = None) -> dict[int, float]:
    """player_id -> age today (not age at a season's Feb 1 like
    `projections.player_ages` — these players are display-only, so "how old is
    he right now" is the more honest question, and does not require a season
    string this module otherwise has no reason to take)."""
    import datetime

    year = reference_year or datetime.date.today().year
    rows = conn.execute(
        "SELECT player_id, birth_date FROM nhl_player_bio WHERE birth_date IS NOT NULL"
    ).fetchall()
    ref = pd.Timestamp(year, 2, 1)
    out = {}
    for pid, born in rows:
        try:
            out[int(pid)] = (ref - pd.Timestamp(born)).days / 365.25
        except (ValueError, TypeError):
            continue
    return out
