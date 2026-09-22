"""Say why, in words a person can argue with.

The draft console learned this the hard way: a ranked list nobody can question
is a list nobody trusts on the clock, so every card ended up carrying its
reasons. The in-season output had the same defect and worse, because it is
read in thirty seconds on a phone rather than studied at a desk.

Three rules, taken from what went wrong there:

Lead with the fact that actually decided it. For a lineup that is almost never
the projection - it is that one player's team is playing tonight and the
other's is not. Saying "+6.12" first buries the only thing the reader needs.

Name the alternative. "Start Tuch" invites "instead of whom?", and the answer
is the difference between advice and an instruction.

Never dress a guess as a fact. A goalie at 63% is a guess, the schedule is a
fact, and the wording has to keep them apart.
"""

from __future__ import annotations

import sqlite3
from datetime import date

from puckpilot.season.settings import LeagueRuntime


def _pretty(day: str) -> str:
    try:
        return date.fromisoformat(day).strftime("%a %d %b").replace(" 0", " ")
    except ValueError:
        return day


def matchups_for(conn: sqlite3.Connection, day: str, season: str) -> dict[str, str]:
    """team -> "vs MTL" or "@ NYI", for the day."""
    rows = conn.execute(
        "SELECT home_team, away_team FROM nhl_schedule "
        "WHERE season = ? AND game_type = 2 AND game_date = ?",
        (season, day),
    ).fetchall()
    out: dict[str, str] = {}
    for r in rows:
        out[r["home_team"]] = f"vs {r['away_team']}"
        out[r["away_team"]] = f"@ {r['home_team']}"
    return out


def move_reasons(conn: sqlite3.Connection, runtime: LeagueRuntime, plan) -> dict[str, str]:
    """player_key -> the one sentence that explains his move."""
    season = runtime.nhl_season
    games = matchups_for(conn, plan.date, season)
    by_key = {c.player.player_key: c for c in plan.playing}
    out: dict[str, str] = {}

    for m in plan.moves:
        p = m.player
        cand = by_key.get(p.player_key)
        where = games.get(p.team, "")
        if m.is_bench:
            if p.team not in games:
                out[p.player_key] = f"{p.team} are not playing tonight."
            else:
                out[p.player_key] = f"plays ({where}) but someone worth more needs the slot."
            continue
        if m.from_slot in ("BN", "?"):
            bit = f"plays tonight ({where})" if where else "plays tonight"
            if cand is not None and cand.p_start is not None:
                bit += f", {cand.p_start:.0%} to start"
            if cand is not None and cand.questionable:
                bit += f" - {p.status}, but the slot would otherwise be empty"
            out[p.player_key] = bit + "."
        else:
            out[p.player_key] = (
                f"same player, different slot - frees {m.from_slot} for someone "
                f"who can only play there."
            )
    return out


def plan_story(conn: sqlite3.Connection, runtime: LeagueRuntime, plan) -> list[str]:
    """The day in plain words, before any table."""
    season = runtime.nhl_season
    games = matchups_for(conn, plan.date, season)
    n_games = (
        conn.execute(
            "SELECT COUNT(*) FROM nhl_schedule WHERE season = ? AND game_type = 2 "
            "AND game_date = ?",
            (season, plan.date),
        ).fetchone()[0]
        or 0
    )
    playing = len(plan.playing)
    total = playing + len(plan.idle) + len(plan.out)

    lines = [
        f"{_pretty(plan.date)} - {n_games} NHL games, and {playing} of your {total} "
        f"players are in one."
    ]
    if plan.is_noop:
        if playing == 0:
            lines.append("Nobody you own plays tonight, so there is nothing to set.")
        else:
            lines.append("Nothing to change: everyone who plays is already in a starting slot.")
    else:
        starts = [m for m in plan.moves if m.is_start and m.from_slot in ("BN", "?")]
        benches = [m for m in plan.moves if m.is_bench]
        bits = []
        if starts:
            bits.append(f"start {_names(starts)}")
        if benches:
            bits.append(f"bench {_names(benches)}")
        lines.append(f"Make {len(plan.moves)} change(s): " + ", ".join(bits) + ".")
        lines.append(
            "Almost always this is about who has a game tonight rather than who is "
            "the better player."
        )

    if plan.empty_slots:
        lines.append(
            f"{len(plan.empty_slots)} of your starting slots will score nothing "
            f"tonight ({', '.join(plan.empty_slots)}) - nobody who plays is eligible "
            f"for them. On a {n_games}-game night that is normal."
        )
    if plan.out:
        lines.append(
            "Out: "
            + ", ".join(
                f"{p.name} ({p.injury_note or p.status_full or p.status})" for p in plan.out
            )
            + "."
        )
    if plan.locked:
        lines.append(
            "Already locked (their game has started): "
            + ", ".join(p.name for p in plan.locked)
            + "."
        )
    if plan.lock_utc:
        lines.append(
            f"You can change anything until {plan.deadline()}, when "
            f"{plan.lock_team}'s game starts; each player locks at his own game."
        )
    idle_teams = sorted({p.team for p in plan.idle if p.team not in games})
    if idle_teams:
        lines.append("Idle tonight: " + ", ".join(idle_teams) + ".")
    return lines


def _names(moves) -> str:
    names = [m.player.name for m in moves]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


# -- the week ---------------------------------------------------------------


def week_story(plan, runtime: LeagueRuntime) -> list[str]:
    """What the category table is saying, and what follows from it."""
    close = plan.close()
    gone = tuple(o for o in plan.outlook if not o.reachable)
    ahead = tuple(o for o in plan.outlook if o.verdict == "ahead")

    lines = [
        f"Week {plan.week} against {plan.opponent or 'your opponent'}, {plan.start} to {plan.end}.",
        f"Your players have {plan.our_games} games this week; theirs have "
        f"{plan.their_games}. More games is more of everything that gets counted, "
        f"so that gap matters on its own.",
        "",
        "A head-to-head week is won by taking more of the twelve categories, not "
        "by scoring most overall. So the question is not who is better - it is "
        "which categories are still undecided.",
    ]
    if ahead:
        lines.append(
            f"Comfortably ahead in {len(ahead)}: "
            + ", ".join(o.category.label for o in ahead)
            + ". Nothing to do there; they are already yours on these projections."
        )
    if gone:
        lines.append(
            "Out of reach: "
            + ", ".join(f"{o.category.label} (behind {abs(o.margin):.1f})" for o in gone)
            + ". Further effort here is wasted - not because they do not matter, but "
            "because nothing you can do this week closes that gap."
        )
    behind = tuple(
        o for o in plan.outlook if o.verdict == "behind" and o.reachable and o not in close
    )
    if behind:
        lines.append(
            "Behind but still winnable: "
            + ", ".join(f"{o.category.label} ({o.margin:+.1f})" for o in behind)
            + ". An add could close these, but they are further away than the ones below."
        )
    if close:
        lines.append(
            "In play: "
            + ", ".join(f"{o.category.label} ({o.margin:+.1f})" for o in close)
            + ". These are where the week is decided, and where an add should go."
        )
    else:
        lines.append(
            "Nothing is close enough to swing, so there is no category worth chasing this week."
        )
    if plan.targets:
        lines.append("")
        lines.append(
            f"The {len(plan.targets)} players below are free agents who would move "
            f"those categories. '+N' is what they add for the week, after "
            f"subtracting what the player you drop would have given you."
        )
    return lines


def protocol_story(proto, adds_left: int | None = None) -> list[str]:
    """What a protocol is, and what approving it would actually do."""
    from puckpilot.season.protocol import CHASE, CONCEDE

    give, chase = proto.of(CONCEDE), proto.of(CHASE)
    lines = [
        f"Suggested plan for week {proto.week}"
        + (f" against {proto.opponent}" if proto.opponent else "")
        + ":",
    ]
    if give:
        lines.append(
            "  Stop spending on: "
            + ", ".join(s.category.label for s in give)
            + " - behind by more than a lineup change and an add together could close."
        )
    if chase:
        lines.append(
            "  Go after: "
            + ", ".join(s.category.label for s in chase)
            + " - close enough that one move could take them."
        )
    if not give and not chase:
        lines.append("  Nothing to concede and nothing close - play it straight.")

    lines.append("")
    lines.append(
        "  What approving this does: it steers which free agents get proposed to "
        "you, toward the categories still in play."
    )
    lines.append(
        "  What it does NOT do: change your daily lineup. That was measured and it "
        "made no difference - most weeks everyone with a game fits in a slot, so "
        "there is no choice for a weighting to change."
    )
    if adds_left is not None:
        lines.append(f"  You have {adds_left} acquisition(s) left this week.")
    return lines
