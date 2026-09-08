from __future__ import annotations

import argparse
import sys

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

    path = run_capture(
        out_dir=out_dir,
        cdp_url=args.cdp,
        user_data_dir=profile,
        dom_interval=args.dom_interval,
        duration=args.duration,
        progress=print,
    )
    print(f"\nCapture written to {path}")
    print("Contents: network.jsonl, websocket.jsonl, events.jsonl, dom/, manifest.json")
    print("Git-ignored — it holds live session data.")
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
        "--duration", type=float, default=None, help="Stop after N seconds (default: until Ctrl+C)"
    )
    capture.add_argument("--out", default=None, help="Output root (default data/captures)")
    capture.set_defaults(func=_cmd_draft_capture)

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
