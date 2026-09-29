"""Why an add is worth making, in the terms a person decides it by.

A proposal card that says "+0.58 categories expected, +3.33 starts" is a
verdict without its reasons. Asked what was behind one, the questions were the
right ones: do those starts account for the roster and the open slots, and what
is each player's floor and ceiling? This answers them from the same numbers
the search priced the add with - nothing here is a second opinion.

    week       your empty slots, then day by day: does he fill an empty slot,
               push someone to the bench, or sit on a full one - and which
               games the drop would have had
    odds       every category the swap moves, down as well as up (dropping a
               goalie costs saves and wins even when it wins shots)
    range      each player's likely total this week, the middle 80% of
               outcomes, from the same distributions the odds use
    per_game   their projected lines
    season     where each ranks on the roster for the rest of the season
"""

from __future__ import annotations

import math
from datetime import date as _date

import numpy as np

from puckpilot.engine.lineup import optimize_lineup, slot_instances
from puckpilot.season import calendar

# Skater categories worth a place in the range summary, and their labels.
RANGE_KEYS = ("goals", "assists", "points", "ppp", "sog", "hits", "blocks", "pim")
PER_GAME_KEYS = ("goals", "assists", "ppp", "sog", "hits", "blocks")
LABELS = {
    "goals": "G",
    "assists": "A",
    "points": "P",
    "ppp": "PPP",
    "sog": "SOG",
    "hits": "HIT",
    "blocks": "BLK",
    "pim": "PIM",
    "wins": "W",
    "saves": "SV",
}
# "Likely" is the middle of the distribution: 10th to 90th percentile.
LOW, HIGH = 0.10, 0.90
# The order sections are shown in, with their headings.
SECTIONS = (
    ("week", "This week, day by day"),
    ("odds", "Category odds, before -> after"),
    ("range", "Likely range this week (middle 80%)"),
    ("per_game", "Projected per game"),
    ("season", "Rest of season"),
)


def explain(
    conn,
    runtime,
    days,
    base,
    after,
    cand,
    drop,
    goalie_source,
    values,
    rates,
    model,
    base_odds=None,
    after_odds=None,
    season_left=None,
    prior=(),
) -> dict[str, list[str]]:
    """section -> lines, for one add priced as `after` against `base`.

    `prior` is the adds proposed ahead of this one. The search is greedy, so
    `base` already has them made, and the card has to say so - approving this
    one alone would not leave the roster these numbers describe.
    """
    before = {d: _day_lineup(conn, runtime, d, base, goalie_source, values) for d in days}
    now = {d: _day_lineup(conn, runtime, d, after, goalie_source, values) for d in days}
    out: dict[str, list[str]] = {
        "week": [open_slots_line(runtime, before)]
        + week_lines(before, now, base, after, cand, drop),
        "range": [range_line(cand, now, rates.get(cand.nhl_player_id) or {}, model)],
        "per_game": [per_game_line(cand, rates.get(cand.nhl_player_id) or {})],
    }
    if prior:
        swaps = "; ".join(
            f"{t.player.name}" + (f" for {t.drop.name}" if t.drop is not None else "")
            for t in prior
        )
        out["week"].insert(0, f"Assumes {swaps} is made too - priced on the roster that leaves.")
    if drop is not None:
        out["range"].append(range_line(drop, before, rates.get(drop.nhl_player_id) or {}, model))
        out["per_game"].append(per_game_line(drop, rates.get(drop.nhl_player_id) or {}))
    if base_odds is not None and after_odds is not None:
        out["odds"] = odds_lines(base_odds, after_odds)
    if season_left is not None and days:
        out["season"] = season_lines(cand, drop, base, values, season_left, days[0])
    return {k: out[k] for k, _ in SECTIONS if k in out}


# -- the week ---------------------------------------------------------------


def _day_lineup(conn, runtime, day, players, goalie_source, values) -> tuple[dict, dict, set]:
    """(skater key -> slot, goalie key -> P(start), clubs playing) for one day.

    The same assignment `week.expected_starts` makes: skaters by the lineup
    optimizer, goalies by value times P(start) into as many G slots as there are.
    """
    shape = runtime.shape()
    playing = calendar.teams_playing(conn, day, runtime.nhl_season)
    p_starts = goalie_source.starts(day) if goalie_source else {}
    skaters, goalies = [], []
    for p in players:
        pid = p.nhl_player_id
        if pid is None or getattr(p, "is_out", False) or getattr(p, "on_ir", False):
            continue
        if p.team not in playing:
            continue
        v = values.per_game(pid, day)
        if p.position == "G":
            ps = float(p_starts.get(pid, 0.0))
            goalies.append((ps * v, ps, p.player_key))
        else:
            skaters.append((p.player_key, p.eligible, v))
    g_slots = sum(n for pos, n in shape.slots if pos == "G")
    goalies.sort(key=lambda x: -x[0])
    in_goal = {key: ps for _, ps, key in goalies[:g_slots] if ps > 0}
    return optimize_lineup(skaters, shape), in_goal, playing


def _slot_name(engine_slot: str) -> str:
    return {"L": "LW", "R": "RW", "UTIL": "Util"}.get(engine_slot, engine_slot)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _short(day: str) -> str:
    return _date.fromisoformat(day).strftime("%a %d").replace(" 0", " ")


def open_slots_line(runtime, lineups: dict) -> str:
    """How many starting slots the roster as it stands leaves empty this week."""
    shape = runtime.shape()
    skater_slots = [s for s in slot_instances(shape) if s != "G"]
    g_slots = sum(n for pos, n in shape.slots if pos == "G")
    empty: dict[str, int] = {}
    for sk, in_goal, _ in lineups.values():
        filled: dict[str, int] = {}
        for slot in sk.values():
            filled[slot] = filled.get(slot, 0) + 1
        for slot in skater_slots:
            if filled.get(slot, 0) > 0:
                filled[slot] -= 1
            else:
                empty[slot] = empty.get(slot, 0) + 1
        if g_slots - len(in_goal) > 0:
            empty["G"] = empty.get("G", 0) + g_slots - len(in_goal)
    total = sum(empty.values())
    days = _plural(len(lineups), "day")
    if not total:
        return (
            f"Your lineup has no empty slots over the {days} left - an add must displace someone."
        )
    order = ["C", "L", "R", "D", "UTIL", "G"]
    parts = ", ".join(
        f"{_slot_name(s)} {empty[s]}"
        for s in sorted(empty, key=lambda s: order.index(s) if s in order else 9)
    )
    return (
        f"Your lineup as it stands leaves {total} slot-games empty over the {days} left ({parts})."
    )


def week_lines(before: dict, after: dict, base, roster_after, cand, drop) -> list[str]:
    """One line per remaining day: what the swap changes in that day's lineup."""
    names = {p.player_key: p.name for p in list(base) + list(roster_after)}
    key = cand.player_key
    lines: list[str] = []
    games = in_lineup = 0
    for day in after:
        b_sk, b_g, playing = before[day]
        a_sk, a_g, _ = after[day]
        bits: list[str] = []
        drop_key = drop.player_key if drop is not None else None
        drop_slot = b_sk.get(drop_key) if drop_key else None
        if cand.team in playing:
            games += 1
        if key in a_sk:
            in_lineup += 1
            benched = [names.get(k, k) for k in b_sk if k not in a_sk and k != drop_key]
            where = _slot_name(a_sk[key])
            if benched:
                bits.append(f"{cand.name} starts ({where}); {', '.join(benched)} to the bench")
            elif drop_slot is not None:
                bits.append(f"{cand.name} starts in {drop.name}'s place ({where})")
            else:
                bits.append(f"{cand.name} fills an empty {where}")
        elif key in a_g:
            in_lineup += 1
            bits.append(f"{cand.name} in goal if he starts ({a_g[key]:.0%})")
        elif cand.team in playing:
            bits.append(f"{cand.name} plays but the lineup is full - bench")
        if drop is not None and not (key in a_sk and drop_slot is not None):
            if drop_slot is not None:
                bits.append(f"{drop.name} loses a start at {_slot_name(drop_slot)}")
            elif drop_key in b_g:
                bits.append(f"{drop.name} loses a {b_g[drop_key]:.0%} chance to start in goal")
        if bits:
            lines.append(f"{_short(day)}: " + "; ".join(bits))
    head = f"{cand.name}: {_plural(games, 'game')} left this week, {in_lineup} in your lineup"
    if not lines:
        return [head, "Neither plays again this week."]
    return [head] + lines


def odds_lines(before, after) -> list[str]:
    """Every category whose chance moves by at least a point, largest first."""
    moves = []
    for b in before.cats:
        a = after.of(b.category.key)
        if a is None:
            continue
        d = a.expected - b.expected
        if abs(d) >= 0.01:
            moves.append((abs(d), f"{b.category.label} {b.expected:.0%} -> {a.expected:.0%}"))
    moves.sort(key=lambda x: -x[0])
    return [text for _, text in moves] or ["No category's chance moves by a point."]


# -- the range ----------------------------------------------------------------


def _quantiles(pmf: np.ndarray) -> tuple[int, int]:
    cdf = np.cumsum(pmf)
    return int(np.searchsorted(cdf, LOW)), int(np.searchsorted(cdf, HIGH))


def _mixed(p_starts: list[float], per_start: np.ndarray) -> np.ndarray:
    """Total over games he may or may not start: each game 0 or a start's total."""
    dist = np.array([1.0])
    for p in p_starts:
        game = per_start * p
        game[0] += 1.0 - p
        dist = np.convolve(dist, game)
    return dist


def range_line(player, lineups: dict, rates: dict, model) -> str:
    """ "Rossi, 4 starts: SOG 6-14, P 1-4, ..." - the middle 80% of his week.

    Skaters: games in the lineup, each played with the model's P(play), with
    the same count distributions the odds use. Goalies: each game's P(start),
    and wins and saves as they would come if he did.
    """
    from puckpilot.season.odds import compound_pmf, count_pmf, trials_pmf

    key = player.player_key
    if player.position == "G":
        p_starts = [g[key] for _, g, _ in lineups.values() if key in g]
        n = sum(p_starts)
        who = f"{player.name}, {n:.1f} expected starts"
        if not p_starts:
            return f"{player.name}: no starts left this week"
        w = min(max(rates.get("wins", 0.0), 0.0), 1.0)
        lo, hi = _quantiles(trials_pmf([p * w for p in p_starts]))
        bits = [f"W {lo}-{hi}"]
        sv = rates.get("saves", 0.0)
        if sv > 0:
            lo, hi = _quantiles(_mixed(p_starts, count_pmf(sv, model.phi.get("saves", 1.0))))
            bits.append(f"SV {lo}-{hi}")
        return f"{who}: " + ", ".join(bits)
    starts = sum(1 for sk, _, _ in lineups.values() if key in sk)
    who = f"{player.name}, {_plural(starts, 'start')}"
    if not starts:
        return f"{player.name}: no starts left this week"
    bits = []
    for k in RANGE_KEYS:
        mean = rates.get(k, 0.0) * starts * model.p_play
        if mean < 0.2:
            continue
        pmf = (
            compound_pmf(mean, model.pim_sizes)
            if k == "pim"
            else count_pmf(mean, model.phi.get(k, 1.0))
        )
        lo, hi = _quantiles(pmf)
        bits.append(f"{LABELS[k]} {lo}-{hi}")
    return f"{who}: " + (", ".join(bits) if bits else "little in any category")


def per_game_line(player, rates: dict) -> str:
    keys = ("wins", "saves") if player.position == "G" else PER_GAME_KEYS
    parts = [f"{LABELS[k]} {rates.get(k, 0.0):.2f}" for k in keys if rates.get(k, 0.0) > 0]
    if player.position == "G" and rates.get("shots_against", 0.0) > 0:
        parts.append(f"SV% {rates.get('saves', 0.0) / rates['shots_against']:.3f}")
    return f"{player.name}: " + (", ".join(parts) or "nothing measurable")


# -- the season ---------------------------------------------------------------


def season_lines(cand, drop, roster, values, season_left, day) -> list[str]:
    """Where each would rank on the roster over the rest of the season.

    The same measure the drop was chosen by - value a game times the games his
    club has left - so this is the reason a drop was allowed, not a new one.
    """

    def worth(p) -> float:
        if p.nhl_player_id is None:
            return 0.0
        return values.per_game(p.nhl_player_id, day) * season_left.get(p.team, 0)

    mates = [
        p
        for p in roster
        if p.nhl_player_id is not None
        and not getattr(p, "on_ir", False)
        and p.player_key != cand.player_key
    ]
    ranked = sorted(mates, key=worth, reverse=True)
    after = sorted(
        [p for p in mates if drop is None or p.player_key != drop.player_key] + [cand],
        key=worth,
        reverse=True,
    )
    pos = next(i for i, p in enumerate(after) if p.player_key == cand.player_key) + 1
    out = [f"{cand.name} would rank {pos} of {len(after)} on your roster"]
    if drop is not None:
        n = len(ranked)
        dpos = next((i for i, p in enumerate(ranked) if p.player_key == drop.player_key), n - 1) + 1
        tail = " - your lowest" if dpos == n else ""
        out.append(f"{drop.name} ranks {dpos} of {n}{tail}")
        gap = worth(cand) - worth(drop)
        if not math.isclose(gap, 0.0, abs_tol=1e-9):
            out.append(
                f"Over the season {cand.name} is worth "
                + ("more" if gap > 0 else "less")
                + f" than {drop.name}"
                + ("" if gap > 0 else " - this is a streaming move for the week")
            )
    return out
