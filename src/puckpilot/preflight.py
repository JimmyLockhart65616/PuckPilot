"""`ppilot draft preflight`: everything that would silently corrupt the board,
checked in one command that fails loudly.

Every check here exists because the failure it catches does not announce
itself. A league file that failed to load falls back to generic defaults and
still produces a plausible board. A keeper missing from the list is still on
the board, and a kept star tops the shortlist exactly like an available one. An
ADP key with a typo builds on a proxy. Keepers modelled in the wrong rounds put
the wrong seat on the clock at pick 1 - and every survival probability on screen
is computed from that clock. None of these crash; all of them are wrong for the
whole draft.

So this does not try to be clever. It builds the real draft-night board, the
same way `draft live` does, and states the numbers a human can check in a
glance: which seat picks first, which picks are ours, how many keepers against
how many slots, how old the player map is, which priced players we cannot see.

Statuses: FAIL corrupts the board (exit non-zero), WARN degrades it, INFO is
for eyeballing. Each check is a plain function so it can be tested alone.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"

# Past this the player map's ADP is a different market from tonight's room.
PLAYERMAP_STALE_DAYS = 3
# How deep into Yahoo's pool the map must resolve: comfortably past the number
# of live picks, so every player the room will actually take is mappable.
COVERAGE_DEPTH = 250
COVERAGE_MIN = 0.95


@dataclass
class Check:
    name: str
    status: str
    detail: str
    lines: list[str] = field(default_factory=list)


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    @property
    def text(self) -> str:
        out = ["Draft preflight", "=" * 60]
        for c in self.checks:
            out.append(f"[{c.status}] {c.name}: {c.detail}")
            out += [f"         {line}" for line in c.lines]
        n = {s: sum(1 for c in self.checks if c.status == s) for s in (FAIL, WARN, PASS)}
        out += [
            "",
            f"{n[FAIL]} fail, {n[WARN]} warn, {n[PASS]} pass -> "
            + ("NOT READY: fix every FAIL before the draft" if self.failed else "READY"),
        ]
        return "\n".join(out)


# ---- league -----------------------------------------------------------------


def check_league_file(path, loader) -> tuple[Check, object | None]:
    """The league file must actually load. The CLI's default path falls back to
    generic 12-team settings with only a stderr line - here that is a FAIL."""
    try:
        league = loader(path)
    except Exception as e:  # LeagueConfigError, OSError, TOML errors alike
        return Check("league file", FAIL, f"{path} did not load: {e}"), None
    return Check("league file", PASS, f"{path} -> {league.name}"), league


def check_league_echo(league) -> Check:
    """Echoed for a human, because a config can parse and still be wrong."""
    shape = league.shape
    slots = " ".join(f"{p}{n}" for p, n in shape.slots)
    return Check(
        "league settings",
        INFO,
        f"{shape.n_teams} teams, {league.scoring.upper()}, roster {shape.roster_size}",
        [
            f"slots {slots} util {shape.util_slots} bench {shape.bench_slots}",
            "skater cats " + " ".join(c.label for c in league.skater_cats),
            "goalie cats " + " ".join(c.label for c in league.goalie_cats),
            f"keepers {league.n_keepers}/team for up to {league.keeper_years} years, "
            f"placed in the {league.keeper_placement.upper()} rounds",
            f"schedule {league.regular_weeks} regular weeks, {league.playoff_teams} playoff "
            f"teams, {league.playoff_weeks} playoff weeks",
        ],
    )


def check_roster_rules(league, rules, board, seat: int) -> Check:
    """`DraftRules.mins` are hardcoded, not derived from the league. They are
    right for a C2/L2/R2/D4/G2 league and wrong for any other - so the moment
    they stop matching the league's starting slots, fail."""
    starters = dict(league.shape.slots)
    lines = [f"mins {rules.mins}  caps {rules.caps}"]
    status, detail = PASS, "roster minimums match the league's starting slots"
    if rules.mins != starters:
        status = FAIL
        detail = f"minimums {rules.mins} do not match the league's starting slots {starters}"
    bad_caps = {p: c for p, c in rules.caps.items() if c < rules.mins.get(p, 0)}
    if bad_caps:
        status = FAIL
        lines.append(f"caps below minimums: {bad_caps}")
    picks_left = board.picks_left(seat)
    need = sum(board.needs(seat).values())
    slack = picks_left - need
    lines.append(
        f"seat {seat}: {picks_left} live picks, {need} still owed to minimums -> "
        f"{slack} free pick(s) before the shortlist can be position-forced"
    )
    return Check("roster rules", status, detail, lines)


# ---- keepers -------------------------------------------------------------------


def check_keepers(conn: sqlite3.Connection, league, season: str, universe) -> Check:
    from puckpilot.keepers import resolve_keepers

    names = league.keepers_for_season(season)
    slots = league.n_keepers * league.shape.n_teams
    if not league.n_keepers:
        return Check("keepers", PASS, "league has no keepers")
    if not names:
        return Check(
            "keepers",
            FAIL,
            f"no keepers.by_season.{season} list - the board would SIMULATE a keeper draw",
        )
    res = resolve_keepers(conn, names)
    known = {int(x) for x in universe.ids}
    outside = [n for n, pid in res.resolved.items() if pid not in known]
    lines = []
    status = PASS
    if res.unmatched:
        status = FAIL
        lines.append(f"not found in nhl_players: {', '.join(res.unmatched)}")
    for entry, cands in res.ambiguous.items():
        status = FAIL
        lines.append(f"ambiguous {entry!r}: {'; '.join(cands)} - qualify it, e.g. 'Name (TEAM)'")
    if outside:
        status = FAIL if status == FAIL else WARN
        lines.append(f"outside the ranked universe (cannot be placed): {', '.join(outside)}")
    if len(names) < slots:
        status = FAIL if status == FAIL else WARN
        lines.append(
            f"{slots - len(names)} of {slots} keeper slots undeclared - any keeper missing "
            "from the list is shown as AVAILABLE"
        )
    return Check(
        "keepers",
        status,
        f"{len(res.resolved)}/{len(names)} resolved, {len(names)} listed for {slots} slots",
        lines,
    )


def check_keeper_owners(conn: sqlite3.Connection, league, season: str, seat: int) -> Check:
    from puckpilot.keepers import resolve_keepers

    if not league.n_keepers:
        return Check("keeper owners", PASS, "league has no keepers")
    owners = league.keeper_owners_for_season(season)
    pool = set(resolve_keepers(conn, league.keepers_for_season(season)).resolved.values())
    lines = []
    status = PASS
    for s, names in sorted(owners.items()):
        res = resolve_keepers(conn, tuple(names))
        stray = [n for n, pid in res.resolved.items() if pid not in pool]
        bad = res.unmatched + list(res.ambiguous)
        if not 0 <= s < league.shape.n_teams:
            status = FAIL
            lines.append(f"seat {s} does not exist in a {league.shape.n_teams}-team league")
        if bad or stray:
            status = FAIL
            lines.append(
                f"seat {s}: unresolvable {bad or '-'}; not in the keeper list {stray or '-'}"
            )
        if len(names) > league.n_keepers:
            status = FAIL
            lines.append(
                f"seat {s} declares {len(names)} keepers; the league allows {league.n_keepers}"
            )
    mine = owners.get(seat)
    if mine is None:
        status = FAIL
        detail = (
            f"seat {seat} (yours) has no declared keepers - your roster panel, needs and "
            "picks left would be dealt at random"
        )
    else:
        detail = f"seat {seat} keeps {', '.join(mine) or 'nobody'}; {len(owners)} seat(s) declared"
        if len(mine) < league.n_keepers:
            status = FAIL if status == FAIL else WARN
            lines.append(
                f"seat {seat} declares {len(mine)} of {league.n_keepers} - any further keeper is "
                "modelled as a live pick"
            )
    return Check("keeper owners", status, detail, lines)


def check_keeper_history(league, season: str, saved: dict | None) -> Check:
    """The keeper list against what the league's own draft history says.

    Offline: reads what `ppilot yahoo keepers` saved. Three disagreements are
    worth a human's minute before the draft - a contract missing from the
    list, a listed name with no contract, and, the one that costs the most, an
    open keeper slot whose likeliest keep is still shown as available.
    """
    from puckpilot.keepers import _norm, split_qualifier

    name = "keeper history"
    if not league.n_keepers:
        return Check(name, PASS, "league has no keepers")
    if not saved or saved.get("season") != season:
        return Check(
            name,
            INFO,
            f"no saved history for {season} - run `ppilot yahoo keepers` to cross-check the list",
        )
    listed = {_norm(split_qualifier(n)[0]) for n in league.keepers_for_season(season)}
    lines: list[str] = []
    status = PASS
    continuing_norms: set[str] = set()
    for m in saved.get("managers", []):
        for p in m.get("continuing", []):
            continuing_norms.add(_norm(p["name"]))
            if _norm(p["name"]) not in listed:
                status = WARN
                lines.append(
                    f"{p['name']} ({m['nickname']}, kept {p['times_kept']}x) has a contract "
                    "but is NOT in the keeper list - shown as available"
                )
        open_slots = int(m.get("open_slots") or 0)
        if not open_slots:
            continue
        cands = m.get("candidates", [])
        declared = [c for c in cands if _norm(c["name"]) in listed]
        still_open = open_slots - len(declared)
        if still_open > 0:
            status = WARN
            priced = [c for c in cands if c.get("adp") is not None][: 4 * still_open]
            likely = ", ".join(f"{c['name']} (ADP {c['adp']})" for c in priced)
            lines.append(
                f"{m['nickname']} (seat {m.get('seat', '?')}): {still_open} undeclared keeper "
                f"slot(s); likeliest, still shown as available: {likely or 'unknown'}"
            )
    # A first-year keep is only possible for a manager with a slot to put him
    # in: a listed name on a full team's roster is a list error, not a keep.
    open_options = {
        _norm(c["name"])
        for m in saved.get("managers", [])
        if int(m.get("open_slots") or 0)
        for c in m.get("candidates", [])
    }
    stray = [
        n
        for n in league.keepers_for_season(season)
        if _norm(split_qualifier(n)[0]) not in continuing_norms
        and _norm(split_qualifier(n)[0]) not in open_options
    ]
    if stray:
        status = WARN
        lines.append(
            "listed, but neither a contract nor a first-year option for a team with an open "
            "slot: " + ", ".join(stray)
        )
    n_cont = len(continuing_norms)
    return Check(
        name,
        status,
        f"{n_cont} continuing contracts in Yahoo history (saved {saved.get('derived_at')})",
        lines,
    )


# ---- the pick sequence ----------------------------------------------------------


def check_pick_sequence(board, league, seat: int) -> Check:
    """The check that would have caught keepers modelled in the wrong rounds:
    the board opened with seat 5 on the clock and our first pick at #16."""
    n = league.shape.n_teams
    full = n * league.shape.roster_size
    kept = len(board.keeper_picks) + len(board.unmatched_keepers)
    ours = [board.slot_numbers[i] for i, (_r, s) in enumerate(board.slots) if s == seat]
    lines = [
        f"{len(board.slots)} live picks ({full} minus {kept} keepers)",
        "your picks (room numbering): " + ", ".join(f"#{p}" for p in ours),
    ]
    status, detail = (
        PASS,
        f"seat {board.slots[0][1]} opens the draft at pick #{board.slot_numbers[0]}",
    )
    if len(board.slots) != full - kept:
        status = FAIL
        detail = f"{len(board.slots)} live picks, expected {full - kept}"
    elif board.keeper_placement == "last" and (
        board.slots[0][1] != 0 or board.slot_numbers[0] != 1
    ):
        status = FAIL
        detail = (
            f"keepers are placed LAST but the draft opens with seat {board.slots[0][1]} "
            f"at #{board.slot_numbers[0]} - expected seat 0 at #1"
        )
    elif board.keeper_placement == "first":
        status = WARN
        lines.append(
            "keepers placed FIRST: early rounds skip seats that keep. Confirm this is how "
            "your league slots keepers; most free-keeper leagues use 'last'."
        )
    if ours and board.keeper_placement == "last" and ours[0] != seat + 1:
        status = FAIL
        lines.append(f"your first pick should be #{seat + 1} in a plain snake")
    return Check("pick order", status, detail, lines)


# ---- Yahoo data ----------------------------------------------------------------


def _age_days(stamp: str | None, now: dt.datetime) -> float | None:
    if not stamp:
        return None
    try:
        then = dt.datetime.fromisoformat(str(stamp).replace(" ", "T"))
    except ValueError:
        return None
    return (now - then).total_seconds() / 86400


def check_playermap(conn: sqlite3.Connection, league_key: str | None, now: dt.datetime) -> Check:
    """The websocket sends Yahoo ids; this map is the only bridge to our board."""
    if not league_key:
        return Check(
            "player map", FAIL, "no league key - cannot check the map picks resolve through"
        )
    row = conn.execute(
        "SELECT COUNT(*), SUM(nhl_player_id IS NOT NULL), MAX(updated_at)"
        " FROM yahoo_player_map WHERE league_key = ?",
        (league_key,),
    ).fetchone()
    total, matched, updated = int(row[0] or 0), int(row[1] or 0), row[2]
    if not total:
        return Check("player map", FAIL, f"empty for {league_key} - run `ppilot yahoo playermap`")
    top = conn.execute(
        "SELECT full_name, nhl_player_id FROM yahoo_player_map WHERE league_key = ?"
        " AND adp_rank <= ? ORDER BY adp_rank",
        (league_key, COVERAGE_DEPTH),
    ).fetchall()
    missing = [str(r[0]) for r in top if r[1] is None]
    cover = 1 - len(missing) / len(top) if top else 0.0
    age = _age_days(updated, now)
    lines = [f"top {COVERAGE_DEPTH} by ADP resolved: {cover:.1%}"]
    if missing:
        lines.append("unresolved in that range: " + ", ".join(missing[:12]))
    status = PASS
    if cover < COVERAGE_MIN:
        status = WARN
    if age is not None and age > PLAYERMAP_STALE_DAYS:
        status = WARN
        lines.append(
            f"built {age:.1f} days ago - ADP has moved since; re-run "
            f"`ppilot yahoo playermap --league-key {league_key}`"
        )
    return Check(
        "player map",
        status,
        f"{matched}/{total} matched, updated {updated}",
        lines,
    )


def check_adp(conn: sqlite3.Connection, explicit: str | None) -> tuple[Check, str | None]:
    from puckpilot.yahoo.playermap import load_adp, resolve_adp_key

    key, notes = resolve_adp_key(conn, explicit, None)
    if key is None:
        return Check("Yahoo ADP", FAIL, "no ADP - the board would run on a proxy", notes), None
    n = len(load_adp(conn, key))
    return Check("Yahoo ADP", PASS, f"{n} players priced from {key}", notes), key


def check_projection_coverage(conn: sqlite3.Connection, league_key: str | None, board) -> Check:
    """Priced players the room will draft that we cannot project. A market-only
    row can still be recorded; a player absent from the board entirely becomes
    an unidentified pick."""
    if not league_key:
        return Check("projection coverage", WARN, "no league key - skipped")
    depth = len(board.slots) + 30
    rows = conn.execute(
        "SELECT full_name, nhl_player_id, adp_rank FROM yahoo_player_map WHERE league_key = ?"
        " AND adp_rank <= ? ORDER BY adp_rank",
        (league_key, depth),
    ).fetchall()
    on_board = {int(pid): str(src) for pid, src in zip(board.u.ids, board.u.source, strict=True)}
    kept = {p.player_id for p in board.keeper_picks}
    market, absent = [], []
    for name, pid, rank in rows:
        if pid is not None and int(pid) in kept:
            continue
        where = on_board.get(int(pid)) if pid is not None else None
        if where == "market":
            market.append(f"{name} ({rank})")
        elif where is None:
            absent.append(f"{name} ({rank})")
    lines = []
    if market:
        lines.append(f"market-priced only, no projection ({len(market)}): " + ", ".join(market))
    if absent:
        lines.append(f"NOT ON THE BOARD at all ({len(absent)}): " + ", ".join(absent))
    status = WARN if absent else PASS
    return Check(
        "projection coverage",
        status,
        f"top {depth} priced: {len(market)} market-only, {len(absent)} absent",
        lines,
    )


# ---- local data ---------------------------------------------------------------


def check_data_freshness(conn: sqlite3.Connection, season: str, now: dt.datetime) -> Check:
    from puckpilot.data.sync import ROSTERS_META

    newest = conn.execute("SELECT MAX(updated_at) FROM sync_meta").fetchone()[0]
    games = conn.execute(
        "SELECT COUNT(*) FROM nhl_schedule WHERE season = ? AND game_type = 2", (season,)
    ).fetchone()[0]
    rosters = conn.execute(
        "SELECT updated_at FROM sync_meta WHERE key = ?", (ROSTERS_META.format(season=season),)
    ).fetchone()
    lines = [f"newest sync record {newest}", f"{games} regular-season games scheduled for {season}"]
    status = PASS
    if not games:
        status = FAIL
        lines.append(
            f"no {season} schedule - `ppilot data sync --schedule-only --seasons {season}`"
        )
    if rosters is None:
        status = FAIL if status == FAIL else WARN
        lines.append(
            f"no record of `ppilot data rosters --season {season}` - player teams (and so "
            "half of every goalie's projected wins) may be stale. Run it."
        )
    else:
        age = _age_days(rosters[0], now)
        lines.append(f"rosters synced {rosters[0]}" + (f" ({age:.1f} days ago)" if age else ""))
        if age is not None and age > PLAYERMAP_STALE_DAYS:
            status = FAIL if status == FAIL else WARN
    return Check("local data", status, f"schedule + roster sync for {season}", lines)


def check_yahoo_probe(prober: Callable[[], object] | None) -> Check:
    """Which Yahoo path is live. Informational: the draft runs on the browser
    session either way."""
    if prober is None:
        return Check("Yahoo OAuth probe", INFO, "skipped (--offline)")
    try:
        result = prober()
    except Exception as e:
        return Check("Yahoo OAuth probe", INFO, f"could not run: {e.__class__.__name__}: {e}")
    verdict = next(
        (line for line in str(result.text).splitlines() if line.startswith("VERDICT")),
        "no verdict",
    )
    return Check("Yahoo OAuth probe", INFO, verdict)


# ---- the whole run ----------------------------------------------------------


def run_preflight(
    conn: sqlite3.Connection,
    league_path,
    seat: int,
    season: str = "20262027",
    adp_league_key: str | None = None,
    prober: Callable[[], object] | None = None,
    now: dt.datetime | None = None,
    progress: Callable[[str], None] = lambda _m: None,
    keeper_history: dict | None = None,
) -> tuple[PreflightReport, object | None]:
    """Every check, in the order a failure would matter. Returns the report
    and the board it built (None if the league did not load)."""
    from puckpilot.draft.live import build_live_board
    from puckpilot.league import load_league
    from puckpilot.yahoo.playermap import load_adp

    now = now or dt.datetime.now(dt.UTC).replace(tzinfo=None)
    report = PreflightReport()
    check, league = check_league_file(league_path, load_league)
    report.checks.append(check)
    if league is None:
        return report, None
    report.checks.append(check_league_echo(league))
    if not 0 <= seat < league.shape.n_teams:
        report.checks.append(
            Check("seat", FAIL, f"seat {seat} outside 0..{league.shape.n_teams - 1}")
        )
        return report, None

    adp_check, key = check_adp(conn, adp_league_key)
    adp = load_adp(conn, key) if key else None
    board = build_live_board(
        conn, league, seat, season=season, adp=adp, league_key=key, progress=progress
    )

    report.checks.append(check_pick_sequence(board, league, seat))
    report.checks.append(check_keepers(conn, league, season, board.u))
    report.checks.append(check_keeper_owners(conn, league, season, seat))
    report.checks.append(check_keeper_history(league, season, keeper_history))
    report.checks.append(check_roster_rules(league, board.rules, board, seat))
    report.checks.append(adp_check)
    report.checks.append(check_playermap(conn, key, now))
    report.checks.append(check_projection_coverage(conn, key, board))
    report.checks.append(check_data_freshness(conn, season, now))
    report.checks.append(check_yahoo_probe(prober))
    return report, board
