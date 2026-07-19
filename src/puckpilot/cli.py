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


def _cmd_data_init(args: argparse.Namespace) -> int:
    from puckpilot.data import store

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    store.init_db(conn)
    print(f"Database initialized: {settings.resolved_db_path}")
    print(f"Tables: {', '.join(sorted(store.table_names(conn)))}")
    return 0


DEFAULT_SEASONS = ["20232024", "20242025", "20252026", "20262027"]
TRAIN_SEASONS = ["20252026", "20242025", "20232024"]  # most recent first


def _cmd_rank(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.engine import projections, validate
    from puckpilot.engine.valuation import rank_players

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)

    if args.validate:
        metrics, text = validate.walk_forward(conn)
        print(text)
        return 0 if metrics["passed"] else 1

    skaters, goalies = projections.project(conn, args.season, TRAIN_SEASONS)
    ranked = rank_players(skaters, goalies)
    print(f"Projected {args.season} ranks (top {args.top}, default categories, 12-team shape):")
    print(f"{'#':>3} {'Name':<26} {'Tm':<4}{'Pos':<4}{'GP':>5} {'VORP':>6}  Projected line")
    for i, (_, r) in enumerate(ranked.head(args.top).iterrows(), start=1):
        if r["kind"] == "skater":
            line = (
                f"G {r['goals']:.0f} A {r['assists']:.0f} +/- {r['plus_minus']:.0f} "
                f"PIM {r['pim']:.0f} PPP {r['ppp']:.0f} SOG {r['sog']:.0f}"
            )
        else:
            line = (
                f"W {r['wins']:.0f} SHO {r['shutouts']:.0f} "
                f"GAA {r['gaa']:.2f} SV% {r['save_pct']:.3f}"
            )
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
    report = run_sims(conn, args.n, seed=args.seed, progress=print)
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
        progress=print,
    )
    print(report.text)
    return 0


def _cmd_waivers_backtest(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.engine.waivers import waiver_backtest

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    report = waiver_backtest(conn, n_teams_tested=args.teams, seed=args.seed, progress=print)
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

    print(f"Syncing schedules: {', '.join(args.seasons)}")
    sync.sync_schedules(conn, nhl, args.seasons, progress=print)

    if not args.schedule_only:
        mp = MoneyPuckClient(cache_dir=settings.resolved_cache_dir / "moneypuck")
        print("Syncing players and game logs" if not args.no_logs else "Syncing players")
        sync.sync_players_and_logs(
            conn, nhl, mp, args.seasons, with_logs=not args.no_logs, progress=print
        )

    print(f"Done: {settings.resolved_db_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppilot", description="PuckPilot fantasy hockey manager")
    sub = parser.add_subparsers(dest="command", required=True)

    league = sub.add_parser("league", help="Yahoo league commands")
    league_sub = league.add_subparsers(dest="subcommand", required=True)
    show = league_sub.add_parser("show", help="Show league settings, categories, and roster")
    show.add_argument("--league-id", default=None, help="Override YAHOO_LEAGUE_ID")
    show.set_defaults(func=_cmd_league_show)

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
    sim.set_defaults(func=_cmd_draft_sim)

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
