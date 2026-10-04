"""Gate G2 for adds: which way of choosing pickups takes more categories?

Replays a season with the live add search itself - `week.build_week_plan`, the
same function the scheduled run calls - fed through roster adapters, and
scores each way of choosing adds by what it exists to raise: categories won
per week against the real schedule of opponents.

    none           keep the drafted roster all season
    share-weekly   search once a week, priced by the share of each gap closed
    share-daily    the same pricing, searched every morning
    odds-weekly    searched once a week, priced by expected categories won
    odds-daily     searched every morning, priced by expected categories won
    goalie-odds    no adds; tonight's goalies chosen by expected categories
                   (`odds.choose_goalies`) instead of starting whoever plays

Any search arm takes a `-s<n>` suffix - `odds-weekly-s3` - to let the bottom
`n` players by rest-of-season value rotate for a streamer instead of the run's
`stream_spots`. Asked after week 1 of 2026-27, when 30-46 empty slot-games went
with two adds proposed and the third acquisition unused.

And a `-p<n>` suffix - `odds-weekly-p1` - searches on each week's last `n` days
against the NEXT week (its opponent, its schedule), spending what is left of
this week's acquisitions: the live run's `preload_days`.

Asked for after week 1, and tested the same way:

    -q<n>    the same search, but a move whose drop plays that day waits for the
             rosters to unlock after the day's games (and so counts against
             the next week's acquisitions); one whose drop is idle is made at
             once from what is left of this week's
    -h       a MID-WEEK move must pay over the rest of this week and the next
             one together (daily arms; Monday's search is unchanged)
    -f<nn>   a mid-week move must clear nn hundredths of a category, not the
             usual floor (daily arms)

`versus` compares every arm with one named arm - the live baseline.

Every input is as-of the morning it is used: the value model and the goalie
start model are clamped to that date (`AsOfValues`, `AsOfGoalieSource`), a
player who has missed two straight team games is out until he plays again,
and the banked part of a week is the real production of whoever the lineup
policy started. The other eleven teams are frozen - nobody else picks anyone
up - so every arm's gain is an upper bound, shared alike, and the arms are
compared with each other rather than with zero. The weekly cap is the
league's; nothing is held back for playoffs (this is the regular season).
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from puckpilot.engine.categories import CATALOG
from puckpilot.league import LeagueConfig
from puckpilot.season.calibration import _components, _result, drop_known_absences

ARMS = ("none", "share-weekly", "share-daily", "odds-weekly", "odds-daily", "goalie-odds")


@dataclass
class ArmSpec:
    """One arm: a base policy, and what its suffixes change about it."""

    kind: str
    spots: int = 2  # -sN: the bottom N by rest-of-season value may be dropped
    preload: int = 0  # -pN: last N days search next week; moves made at once
    # -qN: last N days search next week; a move whose drop plays that day waits
    # for the rosters to unlock after the day's games (and counts next week).
    queue: int = 0
    horizon: bool = False  # -h: a mid-week move must pay over this week's rest + next week
    mid_floor: float | None = None  # -fNN: mid-week bar, in hundredths of a category


def _swap(roster: list[int], target) -> None:
    if target.drop is not None:
        roster.remove(int(target.drop.player_key))
    roster.append(int(target.player.player_key))


def _pair(target) -> tuple[str, str]:
    return (target.player.player_key, target.drop.player_key if target.drop else "")


def _worth(target) -> float:
    return target.gain if target.gain is not None else target.score


def parse_arm(arm: str, stream_spots: int = 2) -> ArmSpec:
    """ "odds-daily-h-f25" -> odds-daily, horizon on, mid-week floor 0.25."""
    base = next(
        (b for b in sorted(ARMS, key=len, reverse=True) if arm == b or arm.startswith(b + "-")),
        None,
    )
    if base is None:
        raise ValueError(f"unknown arm {arm!r}")
    spec = ArmSpec(kind=base, spots=stream_spots)
    for tok in arm[len(base) :].split("-")[1:]:
        if tok == "h":
            spec.horizon = True
        elif len(tok) > 1 and tok[0] in "spqf" and tok[1:].isdigit():
            n = int(tok[1:])
            if tok[0] == "s":
                spec.spots = n
            elif tok[0] == "p":
                spec.preload = n
            elif tok[0] == "q":
                spec.queue = n
            else:
                spec.mid_floor = n / 100
        else:
            raise ValueError(f"unknown option {tok!r} in arm {arm!r}")
    return spec


# Free agents offered to the search each morning, best projections first - the
# live run reads Yahoo's top 150 by the same measure.
POOL_SIZE = 150


class AsOfValues:
    """A value model that cannot see past `cutoff`.

    Live, a later date's form window cannot contain games not yet played. In a
    replay it can: asked about Saturday on Wednesday, it would blend in
    Thursday's and Friday's games. Every question is answered as of the cutoff.
    """

    def __init__(self, values, cutoff: str):
        self.values = values
        self.cutoff = cutoff

    def per_game(self, pid: int, date: str) -> float:
        return self.values.per_game(pid, min(date, self.cutoff))

    def per_game_tilted(self, pid: int, date: str, weights) -> float:
        return self.values.per_game_tilted(pid, min(date, self.cutoff), weights)

    def knows(self, pid: int) -> bool:
        return self.values.knows(pid)

    def projected(self, pid: int) -> float:
        return self.values.projected(pid)


def as_labels(comp: dict[str, float]) -> dict[str, float]:
    """Component totals as the Yahoo labels `build_week_plan` takes banked."""
    by_key = {c.key: c.label for c in CATALOG.values()}
    out = {by_key[k]: v for k, v in comp.items() if k in by_key}
    if "goals_against" in comp:
        out["GA"] = comp["goals_against"]
        hours = comp.get("toi_hours", 0.0)
        if hours > 0:
            out["GAA"] = comp["goals_against"] / hours
    return out


@dataclass
class AddGateReport:
    season: str
    arms: dict[str, list[float]]  # arm -> categories won, per tested team-week
    adds: dict[str, int]  # arm -> acquisitions made
    text: str

    def mean(self, arm: str) -> float:
        v = self.arms[arm]
        return sum(v) / len(v) if v else float("nan")

    def paired(self, a: str, b: str) -> tuple[float, float]:
        """Mean and standard error of (a - b) over the same team-weeks."""
        d = np.array(self.arms[a]) - np.array(self.arms[b])
        if len(d) < 2:
            return float(d.mean()) if len(d) else float("nan"), float("nan")
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def add_gate_report(
    conn: sqlite3.Connection,
    league: LeagueConfig,
    season: str = "20252026",
    train_seasons: tuple[str, ...] | None = None,
    n_tested: int = 6,
    seed: int = 20261,
    arms: tuple[str, ...] = ARMS,
    min_gain: float = 0.25,
    min_expected_gain: float = 0.1,
    stream_spots: int = 2,
    progress: Callable[[str], None] | None = None,
    versus: str = "",
) -> AddGateReport:
    from puckpilot.draft.engine import RosterValuePolicy
    from puckpilot.draft.h2h import round_robin_schedule
    from puckpilot.draft.replay import G_WIDTH, build_replay_data
    from puckpilot.draft.sim import _default_opponents, build_universe, keepers_for, run_draft
    from puckpilot.engine.lineup import optimize_lineup
    from puckpilot.engine.lineup_replay import (
        GameValueModel,
        projected_pg_values,
        skater_availability,
    )
    from puckpilot.season.goalies import AsOfGoalieSource, TrailingStartShareSource
    from puckpilot.season.odds import OddsModel, Side, choose_goalies, goalie_game
    from puckpilot.season.pool import PoolPlayer
    from puckpilot.season.replay import POS_TO_YAHOO, runtime_for_replay, team_by_day
    from puckpilot.season.roster import RosterPlayer, TeamRoster
    from puckpilot.season.values import ValueModel
    from puckpilot.season.week import build_week_plan, per_game_rates

    say = progress or (lambda _m: None)
    if train_seasons is None:
        y = int(season[:4])
        train_seasons = tuple(f"{y - i}{y - i + 1}" for i in range(1, 4))
    rules = league.draft_rules()
    shape = rules.shape
    rng = np.random.default_rng(seed)
    skater_keys = [c.key for c in league.skater_cats]
    cats = league.all_cats

    u = build_universe(conn, season, train_seasons, league)
    data = build_replay_data(conn, season, skater_keys)
    vm = GameValueModel(data, set(u.ids.tolist()), league.goalie_cats)
    pg_value = projected_pg_values(u.frame, vm, skater_keys)
    positions = dict(zip(u.ids.tolist(), u.pos.tolist(), strict=True))
    values = ValueModel(
        vm=vm, data=data, proj_pg=pg_value, season=season, scale_season=season, frame=u.frame
    )

    opponents = _default_opponents(rng, league)
    order = rng.permutation(len(opponents))
    bots, oi = [], 0
    for seat in range(shape.n_teams):
        bots.append(RosterValuePolicy() if seat == 0 else opponents[order[oi]])
        oi += 0 if seat == 0 else 1
    keepers = keepers_for(conn, u, season, league, rng)
    rosters = [[int(u.ids[i]) for i in r] for r in run_draft(u, bots, rules, rng, keepers)]
    say(f"{season}: drafted {len(rosters)} teams")

    everyone = set(int(x) for x in u.ids.tolist())
    played = {
        pid: set(data.skater.get(pid, {})) | set(data.goalie.get(pid, {})) for pid in everyone
    }
    avail = drop_known_absences(skater_availability(conn, season, data, everyone), played)
    teams = team_by_day(conn, season, data, everyone)
    policy = TrailingStartShareSource(conn, season, fallback_season=train_seasons[0])
    runtime = runtime_for_replay(league, season, data.dates)
    index = {d: i for i, d in enumerate(data.dates)}

    def lineup(roster: list[int], i: int) -> dict[int, str]:
        """The morning lineup: schedule, projections, P(start). No forcing."""
        p = policy.starts(data.dates[i])
        cands = []
        for pid in roster:
            if positions.get(pid) == "G":
                if p.get(pid, 0.0) > 0:
                    cands.append((pid, "G", p[pid] * pg_value.get(pid, 0.0)))
            elif i in avail.get(pid, ()):
                cands.append((pid, positions.get(pid, "C"), pg_value.get(pid, 0.0)))
        return optimize_lineup(cands, shape)

    rates = per_game_rates(u.frame, cats)
    g_slots = sum(n for pos, n in shape.slots if pos == "G")
    model = OddsModel()

    def projected_side(roster, banked_sk, banked_g, rest, cutoff, tonight_too=True):
        """Banked totals plus the as-of projection of `rest` (date indices)."""
        frozen = AsOfGoalieSource(policy, cutoff)
        skaters: dict[str, float] = {}
        goalies = []
        for j in rest:
            for pid in lineup_skaters(roster, j):
                for key, r in (rates.get(pid) or {}).items():
                    skaters[key] = skaters.get(key, 0.0) + r
            if not tonight_too and j == rest[0]:
                continue
            p = frozen.starts(data.dates[j])
            gs = sorted(
                (
                    (p.get(pid, 0.0) * pg_value.get(pid, 0.0), p.get(pid, 0.0), pid)
                    for pid in roster
                    if positions.get(pid) == "G" and p.get(pid, 0.0) > 0
                ),
                reverse=True,
            )[:g_slots]
            goalies += [goalie_game(rates.get(pid) or {}, ps) for _, ps, pid in gs]
        return Side(
            banked=_components(banked_sk, banked_g, skater_keys), skaters=skaters, goalies=goalies
        )

    def lineup_skaters(roster, i):
        cands = [
            (pid, positions.get(pid, "C"), pg_value.get(pid, 0.0))
            for pid in roster
            if positions.get(pid) != "G" and i in avail.get(pid, ())
        ]
        return optimize_lineup(cands, shape)

    changed = [0]  # goalie nights the chooser departed from starting everyone

    def lineup_goalie_odds(roster, i, rest, sk, g, opp, opp_done):
        """Skaters as usual; tonight's goalies by expected categories won."""
        date = data.dates[i]
        ours = projected_side(roster, sk, g, rest, date, tonight_too=False)
        zero_sk = np.zeros(len(skater_keys))
        zero_g = np.zeros(G_WIDTH)
        theirs = projected_side(
            rosters[opp],
            sk_day[opp, opp_done].sum(0) if opp_done else zero_sk,
            g_day[opp, opp_done].sum(0) if opp_done else zero_g,
            rest,
            date,
        )
        p = policy.starts(date)
        tonight = {
            pid: goalie_game(rates.get(pid) or {}, p[pid])
            for pid in roster
            if positions.get(pid) == "G" and p.get(pid, 0.0) > 0
        }
        choice = choose_goalies(model, cats, ours, tonight, theirs, slots=g_slots)
        plain = frozenset(pid for pid, slot in lineup(roster, i).items() if slot == "G")
        if choice.start != plain:
            changed[0] += 1
        assigned = dict(lineup_skaters(roster, i))
        for pid in choice.start:
            assigned[pid] = "G"
        return assigned

    def realise(assigned, i, sk, g):
        for pid in assigned:
            line = data.skater.get(pid, {}).get(i)
            if line is not None:
                sk += line
                continue
            line = data.goalie.get(pid, {}).get(i)
            if line is not None:
                g += line

    # The frozen league: every team's realised production, day by day.
    n_days = len(data.dates)
    sk_day = np.zeros((len(rosters), n_days, len(skater_keys)))
    g_day = np.zeros((len(rosters), n_days, G_WIDTH))
    for t, roster in enumerate(rosters):
        for i in range(n_days):
            realise(lineup(roster, i), i, sk_day[t, i], g_day[t, i])
    say(f"{season}: frozen league played")

    weeks = [w for w in runtime.weeks if any(d in index for d in w.dates())][: league.regular_weeks]
    week_days = {w.number: [index[d] for d in w.dates() if d in index] for w in weeks}
    schedule = round_robin_schedule(len(rosters), len(weeks))

    plays_on: dict[int, set[str]] = {}
    for r in conn.execute(
        "SELECT game_date, home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = 2",
        (season,),
    ):
        if r[0] in index:
            plays_on.setdefault(index[r[0]], set()).update((r[1], r[2]))

    def out_on(pid: int, i: int) -> bool:
        """Known to be out that morning: his club plays and he is not available."""
        team = teams.get(pid, [""] * n_days)[i]
        return (
            positions.get(pid) != "G"
            and i not in avail.get(pid, ())
            and team in plays_on.get(i, set())
        )

    def rp(pid: int, i: int) -> RosterPlayer:
        pos = POS_TO_YAHOO.get(positions.get(pid, "C"), "C")
        return RosterPlayer(
            player_key=str(pid),
            yahoo_id=str(pid),
            name=str(pid),
            team=teams.get(pid, [""] * n_days)[i],
            primary_position=pos,
            yahoo_eligible=frozenset({pos}),
            selected_slot="BN",
            nhl_player_id=pid,
            status="O" if out_on(pid, i) else "",
        )

    def fa(pid: int, i: int) -> PoolPlayer:
        pos = POS_TO_YAHOO.get(positions.get(pid, "C"), "C")
        return PoolPlayer(
            player_key=str(pid),
            name=str(pid),
            team=teams.get(pid, [""] * n_days)[i],
            primary_position=pos,
            yahoo_eligible=frozenset({pos}),
            nhl_player_id=pid,
            status="O" if out_on(pid, i) else "",
            ownership_type="freeagents",
        )

    frozen_pids = {pid for r in rosters for pid in r}
    by_value = sorted(everyone, key=lambda pid: -pg_value.get(pid, 0.0))

    results: dict[str, list[float]] = {a: [] for a in arms}
    made: dict[str, int] = dict.fromkeys(arms, 0)
    tested = list(range(min(n_tested, len(rosters))))
    for t in tested:
        for arm in arms:
            spec = parse_arm(arm, stream_spots)
            kind, spots = spec.kind, spec.spots
            last_n = max(spec.preload, spec.queue)
            roster = list(rosters[t])
            adds_season = 0
            # Moves queued at the end of a week for the rosters' unlock are made
            # the next morning, so they count against the week that starts then.
            carry = 0
            for wi, w in enumerate(weeks):
                days = week_days[w.number]
                opp = next((b if a == t else a for a, b in schedule[wi] if t in (a, b)), None)
                if opp is None or not days:
                    continue
                adds_week, carry = carry, 0
                sk = np.zeros(len(skater_keys))
                g = np.zeros(G_WIDTH)
                for k, i in enumerate(days):
                    date = data.dates[i]
                    search = kind not in ("none", "goalie-odds") and (
                        kind.endswith("daily") or k == 0
                    )
                    cap_w = runtime.max_weekly_adds or 99
                    # The week's last days: prepare for the next one.
                    nxt_wi = wi + 1
                    queued: list = []
                    ahead = (
                        last_n > 0
                        and k >= len(days) - last_n
                        and k > 0
                        and nxt_wi < len(weeks)
                        and kind not in ("none", "goalie-odds")
                        and (adds_week < cap_w or (spec.queue > 0 and carry < cap_w))
                    )
                    if ahead:
                        nw = weeks[nxt_wi]
                        nopp = next(
                            (b if a == t else a for a, b in schedule[nxt_wi] if t in (a, b)), None
                        )
                        if nopp is not None and week_days.get(nw.number):
                            ours = TeamRoster(
                                league_key=runtime.league_key,
                                team_key="us",
                                date=date,
                                players=tuple(rp(pid, i) for pid in roster),
                            )
                            theirs = TeamRoster(
                                league_key=runtime.league_key,
                                team_key="them",
                                date=date,
                                players=tuple(rp(pid, i) for pid in rosters[nopp]),
                            )
                            taken = frozen_pids | set(roster)
                            pool = [
                                fa(pid, i)
                                for pid in by_value
                                if pid not in taken and not out_on(pid, i)
                            ][:POOL_SIZE]
                            odds_arm = kind.startswith("odds")
                            plan = build_week_plan(
                                conn,
                                runtime,
                                league,
                                nw,
                                "opp",
                                ours,
                                theirs,
                                pool,
                                u.frame,
                                AsOfGoalieSource(policy, date),
                                AsOfValues(values, date),
                                # Queued moves are paid for next week, not now.
                                adds_used_week=0 if spec.queue else adds_week,
                                adds_used_season=adds_season,
                                min_gain=min_gain,
                                max_targets=cap_w,
                                find_targets=True,
                                odds_model=OddsModel() if odds_arm else None,
                                add_scoring="odds" if odds_arm else "share",
                                min_expected_gain=min_expected_gain,
                                playoff_reserve=0,
                                stream_spots=spots,
                                measure_room=False,
                            )
                            for target in plan.targets:
                                drop = target.drop
                                plays = drop is not None and teams.get(
                                    int(drop.player_key), [""] * n_days
                                )[i] in plays_on.get(i, set())
                                if spec.queue and (plays or adds_week >= cap_w):
                                    # His last game of the week comes first.
                                    if carry >= cap_w:
                                        continue
                                    queued.append(target)
                                    carry += 1
                                else:
                                    if adds_week >= cap_w:
                                        break
                                    _swap(roster, target)
                                    adds_week += 1
                                adds_season += 1
                                made[arm] += 1
                        search = False
                    if search and adds_week < cap_w:
                        opp_done = days[:k]
                        ours = TeamRoster(
                            league_key=runtime.league_key,
                            team_key="us",
                            date=date,
                            players=tuple(rp(pid, i) for pid in roster),
                        )
                        theirs = TeamRoster(
                            league_key=runtime.league_key,
                            team_key="them",
                            date=date,
                            players=tuple(rp(pid, i) for pid in rosters[opp]),
                        )
                        taken = frozen_pids | set(roster)
                        pool = [
                            fa(pid, i)
                            for pid in by_value
                            if pid not in taken and not out_on(pid, i)
                        ][:POOL_SIZE]
                        opp_comp = _components(
                            sk_day[opp, opp_done].sum(0)
                            if opp_done
                            else np.zeros(len(skater_keys)),
                            g_day[opp, opp_done].sum(0) if opp_done else np.zeros(G_WIDTH),
                            skater_keys,
                        )
                        odds_arm = kind.startswith("odds")
                        mid = k > 0
                        floor_now = (
                            spec.mid_floor
                            if mid and spec.mid_floor is not None
                            else min_expected_gain
                        )
                        plan = build_week_plan(
                            conn,
                            runtime,
                            league,
                            w,
                            "opp",
                            ours,
                            theirs,
                            pool,
                            u.frame,
                            AsOfGoalieSource(policy, date),
                            AsOfValues(values, date),
                            adds_used_week=adds_week,
                            adds_used_season=adds_season,
                            min_gain=min_gain,
                            max_targets=cap_w,
                            banked_ours=as_labels(_components(sk, g, skater_keys)),
                            banked_theirs=as_labels(opp_comp),
                            from_day=date,
                            find_targets=True,
                            odds_model=OddsModel() if odds_arm else None,
                            add_scoring="odds" if odds_arm else "share",
                            min_expected_gain=floor_now,
                            playoff_reserve=0,
                            stream_spots=spots,
                            measure_room=False,
                        )
                        targets = list(plan.targets)
                        nopp = (
                            next(
                                (b if a == t else a for a, b in schedule[nxt_wi] if t in (a, b)),
                                None,
                            )
                            if nxt_wi < len(weeks)
                            else None
                        )
                        if mid and spec.horizon and targets and nopp is not None:
                            # A mid-week move must pay over the rest of this
                            # week and the next one together.
                            nplan = build_week_plan(
                                conn,
                                runtime,
                                league,
                                weeks[nxt_wi],
                                "opp",
                                ours,
                                TeamRoster(
                                    league_key=runtime.league_key,
                                    team_key="them",
                                    date=date,
                                    players=tuple(rp(pid, i) for pid in rosters[nopp]),
                                ),
                                [],
                                u.frame,
                                AsOfGoalieSource(policy, date),
                                AsOfValues(values, date),
                                find_targets=False,
                                reprice=[(x.player, x.drop) for x in targets],
                                odds_model=OddsModel() if odds_arm else None,
                                add_scoring="odds" if odds_arm else "share",
                                min_gain=-1e9,
                                min_expected_gain=-1e9,
                                playoff_reserve=0,
                                stream_spots=spots,
                                measure_room=False,
                            )
                            later = {_pair(x): _worth(x) for x in (*nplan.targets, *nplan.lapsed)}
                            targets = [
                                x
                                for x in targets
                                if _worth(x) + later.get(_pair(x), 0.0) >= floor_now
                            ]
                        for target in targets:
                            if adds_week >= cap_w:
                                break
                            _swap(roster, target)
                            adds_week += 1
                            adds_season += 1
                            made[arm] += 1
                    if arm == "goalie-odds":
                        assigned = lineup_goalie_odds(roster, i, days[k:], sk, g, opp, days[:k])
                    else:
                        assigned = lineup(roster, i)
                    realise(assigned, i, sk, g)
                    # After the day's games the rosters unlock: queued moves are made.
                    for target in queued:
                        _swap(roster, target)
                ours_final = _components(sk, g, skater_keys)
                theirs_final = _components(
                    sk_day[opp, days].sum(0), g_day[opp, days].sum(0), skater_keys
                )
                results[arm].append(sum(_result(c, ours_final, theirs_final) for c in cats))
            if arm == "goalie-odds":
                made[arm] = changed[0]
            say(f"{season}: team {t + 1}/{len(tested)} {arm}: {made[arm]} adds so far")

    report = AddGateReport(season=season, arms=results, adds=made, text="")
    lines = [
        f"G2 (adds) - {season}: {len(tested)} teams tested x {len(weeks)} weeks, rivals frozen",
        "",
        f"{'arm':14}{'cats/week':>11}{'adds':>7}   (goalie-odds: nights changed)",
    ]
    for arm in arms:
        lines.append(f"{arm:14}{report.mean(arm):>11.3f}{made[arm]:>7}")
    lines.append("")
    pairs = [(a, "none") for a in arms if a != "none" and "none" in arms]
    if "odds-daily" in arms and "share-daily" in arms:
        pairs.append(("odds-daily", "share-daily"))
    if "share-daily" in arms and "share-weekly" in arms:
        pairs.append(("share-daily", "share-weekly"))
    if "odds-daily" in arms and "odds-weekly" in arms:
        pairs.append(("odds-daily", "odds-weekly"))
    if "odds-weekly" in arms and "share-weekly" in arms:
        pairs.append(("odds-weekly", "share-weekly"))
    pairs += [(a, parse_arm(a, stream_spots).kind) for a in arms]
    if versus:
        pairs += [(a, versus) for a in arms]
    pairs = [(a, b) for a, b in dict.fromkeys(pairs) if b in arms and a != b]
    for a, b in pairs:
        m, se = report.paired(a, b)
        # Both ways: a loss clear of noise is a finding too - 2025-26's -0.33
        # +/- 0.07 for the Sunday search was once labelled "within noise".
        if se == se and m > 2 * se:
            verdict = "clears 2 SE"
        elif se == se and m < -2 * se:
            verdict = "WORSE by more than 2 SE"
        else:
            verdict = "within noise"
        lines.append(f"{a} - {b}: {m:+.3f} +/- {se:.3f}  ({verdict})")
    report.text = "\n".join(lines)
    return report
