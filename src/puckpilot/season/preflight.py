"""Is this thing ready to run for real?

The draft had one night and one preflight. A season has 185 of them, so this
answers a slightly different question: not "will it survive tonight" but "is
every input it depends on actually current". Most in-season failures are
staleness rather than breakage - a roster read from last week, a player map
made before the call-ups, a value model built on a season nobody has synced -
and every one of those produces confident, wrong advice rather than an error.

FAIL means the advice would be wrong. WARN means it would be worse than it
should be. Exits non-zero on any FAIL, so it doubles as the gate for a
scheduled job.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"

# A player map older than this has missed a week of call-ups and waiver churn,
# which is exactly the population an in-season add comes from.
MAP_STALE_DAYS = 7
# Settings change rarely, but the week calendar is read from them.
RUNTIME_STALE_DAYS = 14


@dataclass
class Check:
    name: str
    status: str
    detail: str
    lines: list[str] = field(default_factory=list)


@dataclass
class SeasonPreflightReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    @property
    def text(self) -> str:
        out = ["In-season preflight", "=" * 62]
        for c in self.checks:
            out.append(f"[{c.status}] {c.name}: {c.detail}")
            out += [f"         {line}" for line in c.lines]
        n = {s: sum(1 for c in self.checks if c.status == s) for s in (FAIL, WARN, PASS)}
        out += [
            "",
            f"{n[FAIL]} fail, {n[WARN]} warn, {n[PASS]} pass -> "
            + ("NOT READY: fix every FAIL first" if self.failed else "READY"),
        ]
        return "\n".join(out)


def age_days(stamp: str) -> float | None:
    """How long ago a stored timestamp was written, in days.

    Naive stamps are treated as UTC, because that is what they are: the schema
    defaults to SQLite's `datetime('now')`. Reading them as local time made a
    map written minutes ago report as "-0.2 days old", which is the kind of
    number that makes a person stop believing the rest of the line.
    """
    try:
        when = datetime.fromisoformat((stamp or "").replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (datetime.now(UTC) - when).total_seconds() / 86400.0


# Kept for the checks below, which read better with the short name.
_age_days = age_days


# -- the checks -------------------------------------------------------------


def check_manager(manager) -> Check:
    lines = [f"league {manager.league.name} {manager.league_key or '(from the map)'}"]
    if manager.team_key:
        lines.append(f"team {manager.team_key}")
    else:
        lines.append("team discovered from Yahoo at run time")
    lines.append(f"database {manager.resolved_db()}")
    lines.append("can act: " + ("yes" if manager.can_act else "no - view and decide only"))
    return Check("manager", PASS, manager.name, lines)


def check_authority(manager) -> Check:
    auth = manager.authority
    lines = auth.describe().splitlines()
    detail = "lineups autonomous" if auth.lineup.enabled else "recommend only"
    return Check("authority", INFO, detail, [ln for ln in lines if ln.strip()])


def check_runtime(runtime) -> Check:
    if runtime is None:
        return Check(
            "league settings",
            FAIL,
            "not cached - run `ppilot season settings --refresh`",
        )
    age = _age_days(runtime.fetched_at)
    lines = [
        f"{runtime.start_date} -> {runtime.end_date}, weeks "
        f"{runtime.start_week}-{runtime.end_week}",
        f"{runtime.max_weekly_adds} adds a week of {runtime.max_adds}; "
        f"min goalie games {runtime.min_games_played}",
        f"waivers {runtime.waiver_type}, {runtime.waiver_days}-day, "
        f"FAAB {'yes' if runtime.uses_faab else 'no'}",
    ]
    if not runtime.is_daily_lineup:
        lines.append("NOTE: this league does not lock daily; the daily plan assumes it does")
    if age is not None and age > RUNTIME_STALE_DAYS:
        return Check("league settings", WARN, f"fetched {age:.0f} days ago", lines)
    return Check("league settings", PASS, runtime.name, lines)


def check_categories(runtime, league) -> Check:
    """The categories valued here against the ones Yahoo actually scores.

    `leagues/*.toml` is a transcription, and a transcription can be wrong in a
    way nothing downstream notices: this league's file listed SA, which Yahoo
    shows beside save percentage but marks display-only. Every engine then
    valued goalies for a twelfth category that never decides a week. Silently
    mis-valuing players for the wrong league is this tool's worst failure, so a
    mismatch fails rather than warns.
    """
    ours = [c.label for c in league.all_cats]
    if runtime is None or not runtime.stat_categories:
        return Check(
            "categories",
            WARN,
            "Yahoo's scored categories not cached - run `ppilot season settings --refresh`",
            ["valuing: " + " ".join(ours)],
        )
    scored = list(runtime.scored_labels)
    have, want = {x.casefold() for x in ours}, {x.casefold() for x in scored}
    lines = [f"Yahoo scores {len(scored)}: " + " ".join(scored)]
    display = [s.label for s in runtime.stat_categories if not s.scored]
    if display:
        lines.append("display only (not categories): " + " ".join(display))
    if have != want:
        extra = [x for x in ours if x.casefold() not in want]
        missing = [x for x in scored if x.casefold() not in have]
        detail = "the league file disagrees with Yahoo"
        if extra:
            lines.append("valued here but not scored: " + " ".join(extra))
        if missing:
            lines.append("scored but not valued here: " + " ".join(missing))
        lines.append(f"fix the categories in {league.name}'s league file")
        return Check("categories", FAIL, detail, lines)
    return Check("categories", PASS, f"{len(scored)} match Yahoo", lines)


def check_calendar(runtime, day: str) -> Check:
    if runtime is None:
        return Check("week calendar", FAIL, "no settings, so no calendar")
    if not runtime.weeks:
        return Check(
            "week calendar",
            FAIL,
            "never fetched - weeks are not uniform and are never computed",
        )
    try:
        week = runtime.week_of(day)
    except Exception as e:  # noqa: BLE001
        return Check(
            "week calendar",
            FAIL,
            f"{day} is not covered: {e}",
            [f"{len(runtime.weeks)} weeks known, last ends {runtime.weeks[-1].end}"],
        )
    w = runtime.week(week)
    days = len(w.dates())
    lines = [f"{day} is in week {week} ({w.start} -> {w.end}, {days} days)"]
    if days != 7:
        lines.append("a week of other than seven days - exactly why this is fetched, not computed")
    if runtime.weeks[-1].number < runtime.end_week:
        lines.append(
            f"only weeks {runtime.weeks[0].number}-{runtime.weeks[-1].number} are known; "
            f"the league runs to {runtime.end_week} (playoff matchups are not scheduled yet)"
        )
    return Check("week calendar", PASS, f"week {week}", lines)


def check_schedule(conn: sqlite3.Connection, season: str, day: str) -> Check:
    from puckpilot.season import calendar

    try:
        lo, hi = calendar.season_dates(conn, season)
    except Exception as e:  # noqa: BLE001
        return Check("nhl schedule", FAIL, str(e))
    n = conn.execute(
        "SELECT COUNT(*) FROM nhl_schedule WHERE season = ? AND game_type = 2", (season,)
    ).fetchone()[0]
    playing = calendar.teams_playing(conn, day, season)
    lines = [f"{n} games, {lo} -> {hi}", f"{len(playing)} clubs play on {day}"]
    if not (lo <= day <= hi):
        return Check("nhl schedule", WARN, f"{day} is outside the season", lines)
    lock = calendar.first_lock(conn, playing, day, season)
    if lock:
        lines.append(f"first game {lock[1]} ({lock[0]})")
    return Check("nhl schedule", PASS, f"{n} games synced", lines)


def check_data_freshness(conn: sqlite3.Connection, season: str, day: str) -> Check:
    """Are last night's games actually in the database?"""
    from puckpilot.data.sync import players_behind_their_boxscores

    yesterday = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    played = conn.execute(
        "SELECT COUNT(*) FROM nhl_schedule WHERE season = ? AND game_type = 2 "
        "AND game_date < ? AND game_date >= ?",
        (season, day, yesterday),
    ).fetchone()[0]
    have = conn.execute(
        "SELECT COUNT(DISTINCT s.game_id) FROM nhl_schedule s "
        "JOIN nhl_boxscore_stats b ON b.game_id = s.game_id "
        "WHERE s.season = ? AND s.game_type = 2 AND s.game_date < ? AND s.game_date >= ?",
        (season, day, yesterday),
    ).fetchone()[0]
    behind = len(players_behind_their_boxscores(conn, season))
    lines = [
        f"yesterday: {have} of {played} games have boxscores",
        f"{behind} player(s) with logs behind their boxscores",
    ]
    if played and have < played:
        return Check(
            "data freshness",
            FAIL,
            f"{played - have} of yesterday's games not synced - run `ppilot data daily`",
            lines,
        )
    if behind:
        return Check("data freshness", WARN, f"{behind} players behind", lines)
    return Check("data freshness", PASS, "up to date", lines)


def check_player_map(conn: sqlite3.Connection, league_key: str) -> Check:
    row = conn.execute(
        "SELECT COUNT(*) n, SUM(nhl_player_id IS NOT NULL) matched, MAX(updated_at) at "
        "FROM yahoo_player_map WHERE league_key = ?",
        (league_key,),
    ).fetchone()
    if not row or not row[0]:
        return Check(
            "player map",
            FAIL,
            f"no map for {league_key} - run `ppilot yahoo playermap --league-key {league_key}`",
        )
    n, matched, at = int(row[0]), int(row[1] or 0), row[2]
    age = _age_days(at) if at else None
    lines = [f"{matched} of {n} Yahoo players resolve to an NHL id", f"built {at}"]
    if age is not None and age > MAP_STALE_DAYS:
        return Check(
            "player map",
            WARN,
            f"{age:.0f} days old - call-ups and waiver churn are exactly what it will miss",
            lines,
        )
    return Check("player map", PASS, f"{matched}/{n} mapped", lines)


def check_roster(roster) -> Check:
    if roster is None:
        return Check("roster", FAIL, "could not be read from Yahoo")
    lines = [
        f"{len(roster)} players, {len(roster.starters())} in starting slots",
        f"read for {roster.date}",
    ]
    if roster.injured():
        lines.append("out: " + ", ".join(p.label() for p in roster.injured()))
    stuck = getattr(roster, "illegal_ir", lambda: ())()
    if stuck:
        return Check(
            "roster",
            FAIL,
            "illegal IR: " + ", ".join(f"{p.name} ({p.selected_slot})" for p in stuck),
            lines
            + [
                "no longer IR-eligible; Yahoo blocks every add and drop until activated",
                "activate them (drop someone first if the active roster is full)",
            ],
        )
    if roster.unmapped:
        return Check(
            "roster",
            FAIL,
            f"{len(roster.unmapped)} player(s) not in the map: {', '.join(roster.unmapped)}",
            lines + ["run `ppilot data rosters` then `ppilot yahoo playermap --reresolve-only`"],
        )
    return Check("roster", PASS, roster.team_name or roster.team_key, lines)


def check_projections(values, roster) -> Check:
    if roster is None:
        return Check("projections", WARN, "no roster to check against")
    missing = [
        p.name
        for p in roster.players
        if p.nhl_player_id is None or not values.knows(p.nhl_player_id)
    ]
    lines = [
        f"value model scaled on {values.scale_season}",
        f"{len(values.proj_pg)} players priced",
    ]
    if missing:
        return Check(
            "projections",
            WARN,
            f"{len(missing)} rostered player(s) unpriced: {', '.join(missing)}",
            lines + ["they will never be started over a priced player"],
        )
    return Check("projections", PASS, "every rostered player is priced", lines)


def check_goalies(source, roster, day: str) -> Check:
    if roster is None:
        return Check("starting goalies", WARN, "no roster to check against")
    goalies = [p for p in roster.players if p.position == "G"]
    if not goalies:
        return Check("starting goalies", WARN, "no goalies on the roster")
    starts = source.starts(day)
    lines = [f"{p.name}: {starts.get(p.nhl_player_id, 0.0):.0%} to start" for p in goalies]
    lines.append("model accuracy ~62% (2025-26 61.5%, 2024-25 64.8%, 2023-24 62.4%)")
    if not any(p.nhl_player_id in starts for p in goalies):
        return Check(
            "starting goalies",
            WARN,
            "no goalie on this roster is expected to start today",
            lines,
        )
    return Check("starting goalies", PASS, f"{len(goalies)} goalies priced", lines)


def check_page(manager, key: str) -> Check:
    if not manager.page.publishes:
        return Check("phone page", INFO, "not published")
    if not key:
        return Check(
            "phone page",
            WARN,
            "configured but no key - set PUCKPILOT_MANAGER_KEY",
            [manager.page.url],
        )
    from puckpilot.season import publish
    from puckpilot.web.season_relay import build_id

    try:
        got = publish.health(manager.page.url)
    except publish.PublishError as e:
        return Check("phone page", FAIL, str(e), [manager.page.url])
    local = build_id()
    lines = [manager.page.url, f"serving build {got.get('build')} (local {local})"]
    if got.get("build") != local:
        return Check(
            "phone page",
            WARN,
            "the deployed relay is not this checkout - run deploy/season/update.sh",
            lines,
        )
    return Check("phone page", PASS, "live and current", lines)


def check_plan(plan) -> Check:
    if plan is None:
        return Check("today's plan", FAIL, "did not build")
    lines = [m.describe() for m in plan.moves] or ["no changes needed"]
    if plan.lock_utc:
        lines.append(f"first lock {plan.deadline()} ({plan.lock_team})")
    lines.append(plan.authority_reason)
    return Check(
        "today's plan",
        PASS,
        f"{len(plan.moves)} change(s), {plan.gain:+.2f}",
        lines,
    )
