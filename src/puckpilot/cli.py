from __future__ import annotations

import argparse
import contextlib
import sys
import time
from functools import partial

from puckpilot.config import Settings


def _cmd_league_show(args: argparse.Namespace) -> int:
    from puckpilot.yahoo.auth import MissingYahooCredentials, get_oauth_session
    from puckpilot.yahoo.client import YahooClient

    settings = Settings()
    try:
        oauth = get_oauth_session(settings)
    except MissingYahooCredentials as e:
        print(e, file=sys.stderr)
        return 2

    client = YahooClient(oauth, league_id=args.league_id or settings.yahoo_league_id)
    ov = client.league_overview()

    print(f"League:  {ov['name']}  ({ov['league_key']})")
    print(f"Teams:   {ov['num_teams']}   Scoring: {ov['scoring_type']}")
    print(f"My team: {ov['team_key']}")
    print("\nScoring categories:")
    for cat in ov["stat_categories"]:
        print(f"  {cat['display_name']:<8} {cat.get('position_type', '')}")
    print("\nRoster slots:")
    for pos, meta in ov["roster_positions"].items():
        count = meta.get("count", meta) if isinstance(meta, dict) else meta
        print(f"  {pos:<6} x{count}")
    print("\nCurrent roster:")
    for p in ov["roster"]:
        pos = ",".join(p.get("eligible_positions", []))
        print(f"  [{p.get('selected_position', '?'):>3}] {p['name']:<28} {pos}")
    return 0


def _cmd_yahoo_probe(args: argparse.Namespace) -> int:
    from puckpilot.yahoo.probe import probe

    result = probe(Settings())
    print(result.text)
    return 0 if result.scope_granted else 1


def _cmd_yahoo_league(args: argparse.Namespace) -> int:
    """League overview via the browser session, for when OAuth scope is blocked."""
    import datetime
    from pathlib import Path

    from puckpilot.yahoo.session import YahooSession, YahooSessionError

    settings = Settings()
    profile = settings._resolve(Path("secrets/chrome-profile"))
    if not profile.exists():
        print(
            "No logged-in profile yet. Run `ppilot draft capture --profile` "
            "and sign into Yahoo first.",
            file=sys.stderr,
        )
        return 2
    try:
        with YahooSession(profile) as session:
            keys = [args.league_key] if args.league_key else session.league_keys("nhl")
            if not keys:
                print("No NHL leagues found for this account.", file=sys.stderr)
                return 1
            for key in keys:
                meta = session.league_meta(key)
                print(f"League:  {meta.get('name')}  ({key})")
                print(f"Teams:   {meta.get('num_teams')}   Scoring: {meta.get('scoring_type')}")
                if meta.get("draft_time"):
                    when = datetime.datetime.fromtimestamp(int(meta["draft_time"]))
                    print(f"Draft:   {when:%A %Y-%m-%d %H:%M}   status: {meta.get('draft_status')}")
                print()
                print("Teams:")
                for team in session.teams(key):
                    mine = "  <- you" if str(team.get("is_owned_by_current_login")) == "1" else ""
                    pos = team.get("draft_position") or "-"
                    print(
                        f"  {team.get('team_key', ''):<18} {str(team.get('name'))[:26]:<28}"
                        f"draft_pos={pos}{mine}"
                    )
                picks = session.draft_results(key)
                print()
                print(f"Draft results: {len(picks)} picks recorded")
                for p in picks[: args.picks]:
                    print(
                        f"  R{p.get('round', '?'):<3} #{p.get('pick', '?'):<4}"
                        f"{p.get('team_key', ''):<18} {p.get('player_key', '')}"
                    )
    except YahooSessionError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    return 0


def _cmd_yahoo_watch(args: argparse.Namespace) -> int:
    """Poll draftresults through a live draft to see whether it updates live."""
    from pathlib import Path

    from puckpilot.yahoo.session import YahooSession
    from puckpilot.yahoo.watch import watch

    settings = Settings()
    profile = settings._resolve(Path("secrets/chrome-profile"))
    out = settings._resolve(Path("data/captures")) / "draftwatch.json"
    with YahooSession(profile) as session:
        key = args.league_key or (session.league_keys("nhl") or [None])[0]
        if not key:
            print("No NHL league found.", file=sys.stderr)
            return 2
        report = watch(
            session,
            key,
            interval=args.interval,
            duration=args.duration,
            out_path=out,
            progress=print,
        )
    print()
    print(report.text)
    print()
    print(f"Samples written to {out}")
    return 0


def _cmd_draft_capture(args: argparse.Namespace) -> int:
    from pathlib import Path

    from puckpilot.draft.capture import new_session_dir, run_capture

    settings = Settings()
    root = Path(args.out) if args.out else settings._resolve(Path("data/captures"))
    out_dir = new_session_dir(root)

    profile = None
    if args.profile:
        profile = settings._resolve(Path("secrets/chrome-profile"))

    if not args.cdp and not profile:
        print(
            "No browser attach specified. Preferred (uses your own logged-in Chrome):\n"
            "  1. Close Chrome, then start it with:\n"
            '       & "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" '
            "--remote-debugging-port=9222\n"
            "  2. ppilot draft capture --cdp http://localhost:9222\n\n"
            "Or use a dedicated profile (you log in once, it persists):\n"
            "  ppilot draft capture --profile\n",
            file=sys.stderr,
        )
        return 2

    from puckpilot.draft.capture import ProfileInUse

    try:
        path = run_capture(
            out_dir=out_dir,
            cdp_url=args.cdp,
            user_data_dir=profile,
            url=args.url,
            dom_interval=args.dom_interval,
            values_interval=args.values_interval,
            duration=args.duration,
            all_hosts=args.all_hosts,
            progress=print,
        )
    except ProfileInUse as e:
        print(f"\n{e}", file=sys.stderr)
        return 2
    print(f"\nCapture written to {path}")
    print("Contents: network.jsonl, websocket.jsonl, events.jsonl, dom/, manifest.json")
    print("Git-ignored — it holds live session data.")
    return 0


def _cmd_draft_live(args: argparse.Namespace) -> int:
    """The draft-night console.

    The engine never picks here. It ranks, explains both sides of each option,
    and a human decides - `--web` puts that on a second screen alongside the
    full remaining board, the roster, and what the roster still needs.
    """
    from pathlib import Path

    from puckpilot.data import store
    from puckpilot.draft.live import LiveConfig, build_live_board, run_live

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    league = _league(args)

    adp, feed, ctx, pump_fn = None, None, None, None
    if args.yahoo:
        # The websocket carries picks in any Yahoo draft room, mock or real, and
        # is the only source measured at 100% on the picks it can map.
        from playwright.sync_api import sync_playwright

        from puckpilot.draft.wsfeed import WebsocketFeed, load_yahoo_id_map, pump
        from puckpilot.yahoo.playermap import load_adp

        profile = settings._resolve(Path("secrets/chrome-profile"))
        pw = sync_playwright().start()
        ctx = pw.chromium.launch_persistent_context(str(profile), headless=False, channel="chrome")
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        with contextlib.suppress(Exception):
            page.goto(args.room, wait_until="domcontentloaded")
        feed = WebsocketFeed(ctx, load_yahoo_id_map(conn))
        key = args.yahoo if "." in str(args.yahoo) else None
        adp = load_adp(conn, key) if key else None
        pump_fn = partial(pump, ctx)
        print(f"Websocket feed armed ({len(feed.yahoo_to_nhl)} ids). Join your draft room.")

    board = build_live_board(
        conn, league, seat=args.seat, season=args.season, adp=adp, progress=print
    )
    if args.web:
        return _serve_live(board, feed, ctx, args)
    run_live(board, feed=feed, cfg=LiveConfig(top=args.top), pump=pump_fn)
    return 0


def _serve_live(board, feed, ctx, args: argparse.Namespace) -> int:
    """Second-screen web view, pumped from the browser the feed is attached to."""
    import webbrowser

    from puckpilot.draft.wsfeed import pump
    from puckpilot.web.server import LiveState, serve

    state = LiveState(board=board, feed=feed, top=args.top)
    serve(state, port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print()
    print(f"  Draft view: {url}")
    print("  Ctrl+C to stop.")
    print()
    if not args.no_open:
        webbrowser.open(url)

    # The web view must outlive anything the browser does - tabs opening and
    # closing, navigations, the draft room replacing the lobby. Nothing in here
    # is allowed to end the process except Ctrl+C.
    strikes = 0
    try:
        while True:
            try:
                landed = state.pump()
                if landed:
                    print(f"  +{landed} pick(s)  total={board.made}", flush=True)
                strikes = 0
            except Exception as e:
                strikes += 1
                if strikes in (1, 10, 50):
                    print(f"  feed poll failed ({e.__class__.__name__}); still serving", flush=True)
            if ctx is None:
                time.sleep(args.interval)
            else:
                # Must be a Playwright call, not time.sleep: the sync driver only
                # dispatches events (including "a new tab opened") from inside one.
                pump(ctx, args.interval)
    except KeyboardInterrupt:
        print()
        print("stopping")
    if feed is not None:
        st = feed.status()
        print(
            f"{board.made} picks on the board; feed saw {st.get('picks_detected', 0)} "
            f"(gaps {st.get('gaps') or 'none'}, {st.get('unmapped', 0)} unmapped)"
        )
    return 0


def _cmd_draft_farm(args: argparse.Namespace) -> int:
    """Run mock drafts unattended and harvest ADP + pool-coverage data."""
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    from puckpilot.data import store
    from puckpilot.draft.farm import (
        JOIN_TIMEOUT_S,
        LOBBY,
        MAX_RUNS,
        load_all,
        run_one,
        save,
    )
    from puckpilot.draft.wsfeed import load_yahoo_id_map

    def say(msg: str) -> None:
        # An unattended run's stdout is usually a pipe, and CPython block-buffers
        # a pipe: without flushing, a healthy 40-minute harvest looks dead.
        print(msg, flush=True)

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    league = _league(args)
    out_root = settings._resolve(Path("data/mocks"))
    ymap = load_yahoo_id_map(conn)
    say(f"{len(ymap)} Yahoo ids mapped; harvesting to {out_root}")
    say("Read-only: this records what the room broadcasts and never picks for you.")

    if args.from_capture:
        # Replaying a recording costs nobody a seat, so no browser and no cap.
        from puckpilot.draft.farm import harvest_capture

        roots = (
            [Path(args.from_capture)]
            if args.from_capture != "all"
            else sorted(d for d in settings._resolve(Path("data/captures")).glob("*") if d.is_dir())
        )
        banked = 0
        for src in roots:
            result = harvest_capture(src)
            path = save(result, out_root)
            say(f"  {src.name}: {result.summary}  -> {path.name if path else 'nothing to save'}")
            banked += 1 if path else 0
        say("")
        say(f"{banked} draft(s) banked; {len(load_all(out_root))} mock(s) on disk")
        say("Next: ppilot draft calibrate")
        return 0

    runs = max(1, min(args.runs, MAX_RUNS))
    if runs != args.runs:
        say(f"Capping at {MAX_RUNS} runs: each one takes a seat in a room of real people.")

    profile = settings._resolve(Path("secrets/chrome-profile"))
    done = 0
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(str(profile), headless=False, channel="chrome")
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            for run in range(1, runs + 1):
                say("")
                say(f"[{run}/{runs}] opening the mock lobby")
                with contextlib.suppress(Exception):
                    page.goto(LOBBY, wait_until="domcontentloaded")
                say("  join a mock draft in the browser; recording starts automatically")
                result = run_one(
                    ctx,
                    conn,
                    league,
                    ymap,
                    progress=say,
                    join_timeout=args.join_timeout or JOIN_TIMEOUT_S,
                )
                path = save(result, out_root)
                done += 1
                say(f"  {result.summary}  -> {path.name if path else 'not saved'}")
        except KeyboardInterrupt:
            say("")
            say("stopping")
        except Exception as e:
            say("")
            say(f"browser session ended ({e.__class__.__name__})")

    harvested = load_all(out_root)
    say("")
    say(f"{done} run(s) this session; {len(harvested)} mock(s) on disk")
    say("Next: ppilot draft calibrate")
    return 0


def _cmd_draft_calibrate(args: argparse.Namespace) -> int:
    """Fit survival_spread against harvested mock drafts."""
    from pathlib import Path

    from puckpilot.data import store
    from puckpilot.draft.calibrate import calibrate
    from puckpilot.draft.farm import load_all
    from puckpilot.yahoo.playermap import pool_adp

    settings = Settings()
    root = Path(args.mocks) if args.mocks else settings._resolve(Path("data/mocks"))

    # Yahoo's own pool ADP, not the sparse in-draft advice channel: the latter
    # covered 26 of 192 picks in the 2026-09-08 mock, and a draft's own order
    # cannot stand in for the rank it is being measured against.
    conn = store.connect(settings.resolved_db_path)
    adp = pool_adp(conn, args.league_key)
    if not adp:
        print("No Yahoo pool ADP in the player map; run `ppilot yahoo playermap` first.")
        return 2
    print(f"Fitting against {len(adp)} Yahoo ADP ranks.")

    report = calibrate(load_all(root), incumbent=args.incumbent, adp=adp)
    print(report.text)
    return 0


def _cmd_data_init(args: argparse.Namespace) -> int:
    from puckpilot.data import store

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    store.init_db(conn)
    print(f"Database initialized: {settings.resolved_db_path}")
    print(f"Tables: {', '.join(sorted(store.table_names(conn)))}")
    return 0


TRAIN_SEASONS = ["20252026", "20242025", "20232024"]  # most recent first
DEFAULT_SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026", "20262027"]


def _league(args: argparse.Namespace):
    """League config for this invocation: --league wins, else the configured default."""
    from puckpilot.league import DEFAULT_LEAGUE, load_league

    path = getattr(args, "league", None)
    return load_league(path) if path else DEFAULT_LEAGUE


def _cmd_rank(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.engine import projections, validate
    from puckpilot.engine.valuation import rank_players

    league = _league(args)
    settings = Settings()
    conn = store.connect(settings.resolved_db_path)

    if args.validate:
        metrics, text = validate.walk_forward(conn, league=league)
        print(text)
        return 0 if metrics["passed"] else 1

    skaters, goalies = projections.project(conn, args.season, TRAIN_SEASONS)
    ranked = rank_players(
        skaters,
        goalies,
        shape=league.shape,
        skater_cats=league.skater_cats,
        goalie_cats=league.goalie_cats,
    )
    print(
        f"Projected {args.season} ranks (top {args.top}) - {league.name}: "
        f"{league.shape.n_teams} teams, "
        f"{'/'.join(c.label for c in league.all_cats)}"
    )
    print(f"{'#':>3} {'Name':<26} {'Tm':<4}{'Pos':<4}{'GP':>5} {'VORP':>6}  Projected line")
    for i, (_, r) in enumerate(ranked.head(args.top).iterrows(), start=1):
        cats = league.skater_cats if r["kind"] == "skater" else league.goalie_cats
        parts = []
        for c in cats:
            v = float(r.get(c.key) or 0)
            parts.append(f"{c.label} {v:.3f}" if c.rate else f"{c.label} {v:.0f}")
        line = " ".join(parts)
        print(
            f"{i:>3} {r['name']:<26} {r['team'] or '?':<4}{r['position'] or '?':<4}"
            f"{r['proj_gp']:>5.0f} {r['vorp']:>6.2f}  {line}"
        )
    return 0


def _cmd_draft_sim(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.draft.sim import run_sims

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    league = _league(args)
    report = run_sims(
        conn,
        args.n,
        seed=args.seed,
        league=league,
        scoring=args.scoring or league.scoring,
        progress=print,
    )
    print(report.text)
    return 0 if report.passed else 1


def _cmd_draft_mock(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.draft.mock import run_mock

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    result = run_mock(
        conn,
        league=_league(args),
        seat=args.seat,
        seed=args.seed,
        auto=args.auto,
        target_season=args.season,
        progress=print,
    )
    print(result.text)
    return 0


def _cmd_lineup_replay(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.engine.lineup_replay import bench_regret_report

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    report = bench_regret_report(
        conn,
        n_drafts=args.drafts,
        seed=args.seed,
        goalie_accuracy=args.goalie_accuracy,
        league=_league(args),
        progress=print,
    )
    print(report.text)
    return 0


def _cmd_waivers_backtest(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.engine.waivers import waiver_backtest

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    report = waiver_backtest(
        conn, n_teams_tested=args.teams, seed=args.seed, league=_league(args), progress=print
    )
    print(report.text)
    return 0


def _cmd_shadow_season(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.shadow import run_shadow_season

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    report = run_shadow_season(
        conn,
        season=args.season,
        seed=args.seed,
        manage_waivers=not args.no_waivers,
        league=_league(args),
        progress=print,
    )
    print()
    print(report.text)
    return 0


def _cmd_data_sync(args: argparse.Namespace) -> int:
    from puckpilot.data import store, sync
    from puckpilot.data.moneypuck import MoneyPuckClient, season_start_year
    from puckpilot.data.nhl import NhlClient

    for season in args.seasons:
        season_start_year(season)  # raises ValueError on malformed input

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    store.init_db(conn)
    nhl = NhlClient()

    if args.boxscores_only:
        print(f"Syncing boxscores (hits/blocks): {', '.join(args.seasons)}")
        sync.sync_boxscores(conn, nhl, args.seasons, progress=print)
        print(f"Done: {settings.resolved_db_path}")
        return 0

    print(f"Syncing schedules: {', '.join(args.seasons)}")
    sync.sync_schedules(conn, nhl, args.seasons, progress=print)

    if not args.schedule_only:
        mp = MoneyPuckClient(cache_dir=settings.resolved_cache_dir / "moneypuck")
        print("Syncing players and game logs" if not args.no_logs else "Syncing players")
        sync.sync_players_and_logs(
            conn, nhl, mp, args.seasons, with_logs=not args.no_logs, progress=print
        )
        print("Syncing boxscores (hits/blocks)")
        sync.sync_boxscores(conn, nhl, args.seasons, progress=print)
        print("Syncing player bios (birth dates)")
        sync.sync_player_bios(conn, nhl, args.seasons, progress=print)

    print(f"Done: {settings.resolved_db_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppilot", description="PuckPilot fantasy hockey manager")
    parser.add_argument(
        "--league",
        default=None,
        metavar="PATH",
        help="League TOML to use (default: the LEAGUE_FILE setting)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    league = sub.add_parser("league", help="Yahoo league commands")
    league_sub = league.add_subparsers(dest="subcommand", required=True)
    show = league_sub.add_parser("show", help="Show league settings, categories, and roster")
    show.add_argument("--league-id", default=None, help="Override YAHOO_LEAGUE_ID")
    show.set_defaults(func=_cmd_league_show)

    yahoo = sub.add_parser("yahoo", help="Yahoo API diagnostics")
    yahoo_sub = yahoo.add_subparsers(dest="subcommand", required=True)
    probe = yahoo_sub.add_parser(
        "probe", help="Report OAuth status and Fantasy API scope in one command"
    )
    probe.set_defaults(func=_cmd_yahoo_probe)

    yl = yahoo_sub.add_parser(
        "league",
        help="League settings, teams and draft results via the logged-in browser session",
    )
    yl.add_argument("--league-key", default=None, help="e.g. 477.l.29326 (default: auto-detect)")
    yl.add_argument("--picks", type=int, default=10, help="Draft picks to print")
    yl.set_defaults(func=_cmd_yahoo_league)

    yw = yahoo_sub.add_parser(
        "watch-draft",
        help="Poll draftresults through a live draft to prove whether it updates live",
    )
    yw.add_argument("--league-key", default=None, help="e.g. 477.l.29326 (default: auto-detect)")
    yw.add_argument("--interval", type=float, default=3.0, help="Seconds between polls")
    yw.add_argument("--duration", type=float, default=5400.0, help="Give up after N seconds")
    yw.set_defaults(func=_cmd_yahoo_watch)

    data = sub.add_parser("data", help="Local data store commands")
    data_sub = data.add_subparsers(dest="subcommand", required=True)
    init = data_sub.add_parser("init", help="Create/upgrade the local SQLite database")
    init.set_defaults(func=_cmd_data_init)

    sync = data_sub.add_parser("sync", help="Sync NHL schedules, MoneyPuck stats, and game logs")
    sync.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        metavar="YYYYYYYY",
        help=f"Seasons like 20252026 (default: {' '.join(DEFAULT_SEASONS)})",
    )
    sync.add_argument(
        "--schedule-only", action="store_true", help="Sync schedules only, skip players/stats/logs"
    )
    sync.add_argument(
        "--no-logs", action="store_true", help="Sync schedules and MoneyPuck stats, skip game logs"
    )
    sync.add_argument(
        "--boxscores-only",
        action="store_true",
        help="Sync per-game boxscores only (hits/blocks); use for the one-time backfill",
    )
    sync.set_defaults(func=_cmd_data_sync)

    rank = sub.add_parser("rank", help="Project and rank players by category value")
    rank.add_argument("--season", default="20262027", help="Target season (default 20262027)")
    rank.add_argument("--top", type=int, default=30, help="Rows to print (default 30)")
    rank.add_argument(
        "--validate",
        action="store_true",
        help="Walk-forward backtest: train <=2024-25, predict 2025-26, report MAE + Spearman",
    )
    rank.set_defaults(func=_cmd_rank)

    draft = sub.add_parser("draft", help="Draft assistant and simulator")
    draft_sub = draft.add_subparsers(dest="subcommand", required=True)
    sim = draft_sub.add_parser("sim", help="Monte Carlo draft sim vs bots + season replay")
    sim.add_argument("--n", type=int, default=300, help="Number of simulated drafts")
    sim.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    sim.add_argument(
        "--scoring",
        choices=("h2h", "roto"),
        default=None,
        help="h2h plays the league's real weekly matchups + playoffs; roto is the old baseline",
    )
    sim.set_defaults(func=_cmd_draft_sim)

    capture = draft_sub.add_parser(
        "capture",
        help="Record a Yahoo draft room (read-only) to build the pick feed against real evidence",
    )
    capture.add_argument(
        "--cdp",
        default=None,
        metavar="URL",
        help="Attach to your own Chrome started with --remote-debugging-port "
        "(e.g. http://localhost:9222). Preferred: you are already logged in.",
    )
    capture.add_argument(
        "--profile",
        action="store_true",
        help="Launch Chrome with a dedicated persistent profile under secrets/ instead",
    )
    capture.add_argument(
        "--dom-interval", type=float, default=15.0, help="Seconds between DOM snapshots"
    )
    capture.add_argument(
        "--values-interval",
        type=float,
        default=1.5,
        help="Seconds between reads of the header pick counter (default 1.5)",
    )
    capture.add_argument(
        "--duration", type=float, default=None, help="Stop after N seconds (default: until Ctrl+C)"
    )
    capture.add_argument(
        "--url", default=None, help="Open this URL on start (e.g. the Yahoo mock draft lobby)"
    )
    capture.add_argument(
        "--all-hosts",
        action="store_true",
        help="Record every host, not just Yahoo's. A fantasy page is mostly ad "
        "exchanges, so the default keeps the log readable.",
    )
    capture.add_argument("--out", default=None, help="Output root (default data/captures)")
    capture.set_defaults(func=_cmd_draft_capture)

    mock = draft_sub.add_parser(
        "mock",
        help="Draft interactively vs the bot field, then replay the roster over a real season",
    )
    mock.add_argument("--seat", type=int, default=0, help="Your draft slot (0-based)")
    mock.add_argument("--seed", type=int, default=None, help="RNG seed for a repeatable field")
    mock.add_argument(
        "--auto",
        action="store_true",
        help="Let the engine draft your seat too (regression check, no prompting)",
    )
    mock.add_argument(
        "--season",
        default="20252026",
        help="Season to draft for and replay; must be complete to be graded",
    )
    mock.set_defaults(func=_cmd_draft_mock)

    farm = draft_sub.add_parser(
        "farm",
        help="Sit through Yahoo mock drafts (read-only) and harvest ADP / pool-coverage data",
    )
    farm.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Mock drafts to sit through (capped: each run occupies a seat in a room "
        "of real people)",
    )
    farm.add_argument(
        "--join-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="How long to hold the lobby open waiting for you to join a mock "
        "(default: farm.JOIN_TIMEOUT_S, 15 min)",
    )
    farm.add_argument(
        "--from-capture",
        default=None,
        metavar="DIR|all",
        help="Bank an existing `draft capture` recording instead of sitting through a "
        "live mock. No browser, no seat taken. 'all' replays every capture on disk.",
    )
    farm.set_defaults(func=_cmd_draft_farm)

    cal = draft_sub.add_parser("calibrate", help="Fit survival_spread to harvested mock drafts")
    cal.add_argument("--mocks", default=None, help="Harvest dir (default data/mocks)")
    cal.add_argument("--incumbent", type=float, default=6.0, help="Current survival_spread")
    cal.add_argument(
        "--league-key",
        default=None,
        metavar="KEY",
        help="Restrict pool ADP to one league key (default: every mapped league)",
    )
    cal.set_defaults(func=_cmd_draft_calibrate)

    live = draft_sub.add_parser("live", help="Draft-night console: live recommendations")
    live.add_argument("--seat", type=int, default=0, help="Your draft slot (0-based)")
    live.add_argument("--season", default="20262027", help="Season to draft for")
    live.add_argument("--top", type=int, default=12, help="Candidates on screen")
    live.add_argument(
        "--yahoo",
        nargs="?",
        const="auto",
        default=None,
        metavar="LEAGUE_KEY",
        help="Read picks from the draft room websocket (needs a logged-in profile "
        "and a built player map). Optionally name a league key for real ADP.",
    )
    live.add_argument(
        "--room",
        default="https://hockey.fantasysports.yahoo.com/hockey",
        help="Page to open when --yahoo is used",
    )
    live.add_argument(
        "--web",
        action="store_true",
        help="Serve the second-screen view (shortlist + full board + roster) instead "
        "of the terminal console",
    )
    live.add_argument("--port", type=int, default=8765, help="Port for --web")
    live.add_argument("--no-open", action="store_true", help="Do not open a browser tab")
    live.add_argument(
        "--interval", type=float, default=1.0, help="Seconds between feed polls with --web"
    )
    live.set_defaults(func=_cmd_draft_live)

    lineup = sub.add_parser("lineup", help="Daily lineup tools")
    lineup_sub = lineup.add_subparsers(dest="subcommand", required=True)
    replay = lineup_sub.add_parser(
        "replay", help="Bench-regret replay: optimizer vs hindsight vs set-and-forget"
    )
    replay.add_argument("--drafts", type=int, default=2, help="Drafts to source rosters from")
    replay.add_argument("--seed", type=int, default=123)
    replay.add_argument(
        "--goalie-accuracy",
        type=float,
        default=0.9,
        help="Morning goalie announcement accuracy for the noisy source (default 0.9)",
    )
    replay.set_defaults(func=_cmd_lineup_replay)

    shadow = sub.add_parser("shadow", help="End-to-end season simulation")
    shadow_sub = shadow.add_subparsers(dest="subcommand", required=True)
    shadow_season = shadow_sub.add_parser(
        "season", help="Draft + daily lineups + waivers replayed over a real season"
    )
    shadow_season.add_argument("--season", default="20252026", help="Season to replay")
    shadow_season.add_argument("--seed", type=int, default=11)
    shadow_season.add_argument(
        "--no-waivers", action="store_true", help="Draft and set lineups only, no in-season moves"
    )
    shadow_season.set_defaults(func=_cmd_shadow_season)

    waivers = sub.add_parser("waivers", help="Waiver/FA engine tools")
    waivers_sub = waivers.add_subparsers(dest="subcommand", required=True)
    backtest = waivers_sub.add_parser(
        "backtest", help="Weekly add/drop recommendations replayed vs standing pat"
    )
    backtest.add_argument("--teams", type=int, default=6, help="Drafted teams to test")
    backtest.add_argument("--seed", type=int, default=7)
    backtest.set_defaults(func=_cmd_waivers_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
