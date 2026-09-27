from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from functools import partial
from pathlib import Path

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
    from puckpilot.yahoo.watch import DEFAULT_INTERVAL_S, watch

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
            interval=args.interval or DEFAULT_INTERVAL_S,
            duration=args.duration,
            out_path=out,
            progress=print,
        )
    print()
    print(report.text)
    print()
    print(f"Samples written to {out}")
    return 0


def _cmd_data_rosters(args: argparse.Namespace) -> int:
    """Repoint nhl_players.team_abbrev at who each player actually plays for."""
    from puckpilot.data import store, sync
    from puckpilot.data.nhl import NhlClient

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    print(f"Syncing {args.season} rosters from the NHL API (32 requests)...")
    result = sync.sync_current_rosters(conn, NhlClient(), args.season, progress=print)
    print()
    print(f"{result['changed']} of {result['players']} players had the wrong team.")
    if result["changed"]:
        print("Re-run `ppilot rank` - goalie win projections blend on team strength.")
    return 0


def _cmd_yahoo_playermap(args: argparse.Namespace) -> int:
    """Build the Yahoo player-key -> NHL id map, and Yahoo's own ADP with it.

    Two error messages told the user to run this and it did not exist: the map
    is what lets the draft-room websocket (which sends ids, not names) mark
    players off the board, and `draft calibrate` refuses to fit without the ADP
    it carries. The 400 rows on disk were written once, ad hoc.

    `--reresolve-only` skips Yahoo entirely and re-tries every already-fetched
    row that has no NHL id against `nhl_players` as it stands now. Worth
    running on its own after `data sync` picks up new rookies - no browser
    session needed for information Yahoo was never asked to give again.
    """
    from puckpilot.data import store

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)

    if args.reresolve_only:
        from puckpilot.yahoo.playermap import reresolve_unmatched

        report = reresolve_unmatched(conn, progress=print)
        print()
        print(f"Re-resolved {report.matched}/{report.total} previously-unmatched players.")
        if report.fallbacks:
            print("  via fallback matching, verify these:")
            for f in report.fallbacks[:15]:
                print(f"    {f}")
        if report.unmatched:
            shown = ", ".join(report.unmatched[:12])
            more = f" (+{len(report.unmatched) - 12} more)" if len(report.unmatched) > 12 else ""
            print(f"  still unmatched: {shown}{more}")
        return 0

    from pathlib import Path

    from puckpilot.yahoo.playermap import build_map
    from puckpilot.yahoo.session import YahooSession, YahooSessionError

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
            key = args.league_key or (session.league_keys("nhl") or [None])[0]
            if not key:
                print("No NHL league found for this account.", file=sys.stderr)
                return 2
            print(f"Building the player map for {key}...")
            report = build_map(conn, session, key, limit=args.limit, progress=print)
    except YahooSessionError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    print()
    print(report.text)
    return 0


def _cmd_yahoo_keepers(args: argparse.Namespace) -> int:
    """Keeper contracts reconstructed from the league's Yahoo draft history.

    Prints; never writes the league file, which is private and hand-kept.
    """
    from pathlib import Path

    from puckpilot.data import store
    from puckpilot.keepers import _norm
    from puckpilot.yahoo.keeperhistory import derive_contracts, fetch_history
    from puckpilot.yahoo.session import YahooSession, YahooSessionError

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    league = _league(args)
    profile = settings._resolve(Path("secrets/chrome-profile"))
    try:
        with YahooSession(profile) as session:
            key = args.league_key or (session.league_keys("nhl") or [None])[0]
            if not key:
                print("No NHL league found for this account.", file=sys.stderr)
                return 2
            print(f"Reading keeper history for {key}...")
            history, current, _meta = fetch_history(
                session, key, depth=league.keeper_years + 1, progress=print
            )
            # What the league has actually declared, once it has: the source of
            # truth, where the contract history is only a forecast of it.
            declared = session.keepers(key)
    except YahooSessionError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    if not history:
        print("No previous seasons found - nothing to reconstruct contracts from.")
        return 1

    report = derive_contracts(
        history,
        current,
        n_keepers=league.n_keepers,
        keeper_years=league.keeper_years,
        roster_rounds=league.shape.roster_size,
    )

    # Our view of each candidate: projected VORP for the season being drafted
    # and Yahoo's ADP rank, both through the player map (Yahoo ids are stable
    # across seasons, so a bare id finds a player in this season's map).
    bare_to_nhl: dict[str, int] = {}
    bare_adp: dict[str, int] = {}
    for pkey, nhl_id, rank in conn.execute(
        "SELECT player_key, nhl_player_id, adp_rank FROM yahoo_player_map"
    ):
        b = str(pkey).rsplit(".", 1)[-1]
        if nhl_id is not None:
            bare_to_nhl[b] = int(nhl_id)
        if rank is not None:
            bare_adp[b] = int(rank)
    vorp: dict[int, float] = {}
    try:
        from puckpilot.draft.sim import build_universe

        u = build_universe(conn, args.season, tuple(TRAIN_SEASONS), league)
        vorp = {int(p): float(v) for p, v in zip(u.ids, u.vorp, strict=True)}
    except Exception as e:  # projections are a nicety here, not the point
        print(f"  (no projections: {e.__class__.__name__}: {e})")

    def label(b: str) -> str:
        name = report.names.get(b, f"yahoo {b}")
        bits = []
        if bare_to_nhl.get(b) in vorp:
            bits.append(f"VORP {vorp[bare_to_nhl[b]]:+.1f}")
        if b in bare_adp:
            bits.append(f"ADP {bare_adp[b]}")
        return f"{name} ({', '.join(bits)})" if bits else name

    # Seats: Yahoo's own draft order once it is set, else --order.
    order = [s.strip().lower() for s in (args.order or "").split(",") if s.strip()]
    seat_of: dict[str, int] = {}
    for team in current.values():
        pos = team.get("draft_position")
        if str(pos or "").isdigit():
            seat_of[str(team.get("guid"))] = int(pos) - 1
    for m in report.managers:
        if m.guid in seat_of or not order:
            continue
        for i, want in enumerate(order):
            if want in (m.nickname.lower(), m.team_name.lower(), str(m.current_team_key).lower()):
                seat_of[m.guid] = i

    season_label = f"{args.season[:4]}-{args.season[6:]}"
    print()
    print(f"Keeper contracts going into {season_label}, from {', '.join(report.seasons)}")
    print(f"({league.n_keepers} keepers/team, a contract runs {league.keeper_years} keeps)")
    print(
        "Keeper rounds found per season: "
        + ", ".join(f"{k} last {n}" for k, n in report.keeper_rounds.items())
    )
    for m in report.managers:
        seat = seat_of.get(m.guid)
        print()
        print(
            f"{m.nickname} - {m.team_name} ({m.current_team_key or 'not in this season'})"
            f"  seat {seat if seat is not None else '?'}"
        )
        cont = ", ".join(f"{label(b)} kept {n}x" for b, n in m.continuing) or "none"
        print(f"  continuing: {cont}")
        if m.expired:
            print(f"  expired:    {', '.join(report.names.get(b, b) for b in m.expired)}")
        slots = m.open_slots(league.n_keepers)
        if slots:
            ranked = sorted(
                m.candidates,
                key=lambda b: -vorp.get(bare_to_nhl.get(b, -1), float("-inf")),
            )
            top = ", ".join(label(b) for b in ranked[: args.candidates])
            print(f"  {slots} open slot(s); first-year keep candidates by our VORP: {top}")
    if report.lapsed:
        print()
        print(
            "Kept last season but on no roster at its end: "
            + ", ".join(report.names.get(b, b) for b in report.lapsed)
        )

    # Cross-check against the league file, by normalized name.
    listed = {_norm(n.split("(")[0]): n for n in league.keepers_for_season(args.season)}
    continuing = {
        _norm(report.names.get(b, "")): report.names.get(b, b)
        for m in report.managers
        for b, _n in m.continuing
    }
    print()
    print(f"Against keepers.by_season.{args.season} ({len(listed)} listed):")
    missing = [continuing[k] for k in continuing if k not in listed]
    extra = [listed[k] for k in listed if k not in continuing]
    print(f"  continuing contracts missing from the file: {', '.join(missing) or 'none'}")
    print(f"  listed but not a continuing contract:       {', '.join(extra) or 'none'}")
    print("  (a listed name that is not continuing may be a declared first-year keep - check)")

    # Declared keepers, when the league has set them, replace the forecast.
    guid_of_team = {k: str(t.get("guid")) for k, t in current.items()}
    nick_of_team = {k: str(t.get("nickname")) for k, t in current.items()}
    declared_rows = [
        {
            "name": str(p.get("full") or ""),
            "nhl_id": bare_to_nhl.get(str(p.get("player_key", "")).rsplit(".", 1)[-1]),
            "owner_team_key": p.get("owner_team_key"),
            "owner": nick_of_team.get(str(p.get("owner_team_key")), "?"),
            "seat": seat_of.get(guid_of_team.get(str(p.get("owner_team_key")), "")),
        }
        for p in declared
    ]
    print()
    if declared_rows:
        print(f"DECLARED in Yahoo ({len(declared_rows)} keepers) - this is the list to use:")
        by_owner: dict[str, list[dict]] = {}
        for r in declared_rows:
            by_owner.setdefault(f"{r['owner']} (seat {r['seat']})", []).append(r)
        for owner, rows in sorted(by_owner.items()):
            print(f"  {owner}: {', '.join(r['name'] for r in rows)}")
        declared_norm = {_norm(r["name"]) for r in declared_rows}
        print(f"Against keepers.by_season.{args.season}:")
        print(
            "  declared but missing from the file: "
            + (
                ", ".join(r["name"] for r in declared_rows if _norm(r["name"]) not in listed)
                or "none"
            )
        )
        print(
            "  in the file but not declared:       "
            + (", ".join(v for k, v in listed.items() if k not in declared_norm) or "none")
        )
        print()
        print("Suggested TOML (declared keepers):")
        print(f"[keepers.owners.{args.season}]")
        for rows in sorted(
            by_owner.values(), key=lambda rs: 99 if rs[0]["seat"] is None else rs[0]["seat"]
        ):
            seat = rows[0]["seat"]
            names = ", ".join(f'"{r["name"]}"' for r in rows)
            prefix = f'"{seat}"' if seat is not None else "# seat ?"
            print(f"{prefix} = [{names}]  # {rows[0]['owner']}")
    else:
        print("No keepers declared in Yahoo yet.")
        print("Suggested TOML - continuing contracts only; add first-year keeps as declared:")
        print(f"[keepers.owners.{args.season}]")
        for m in sorted(report.managers, key=lambda m: seat_of.get(m.guid, 99)):
            seat = seat_of.get(m.guid)
            names = ", ".join(f'"{report.names.get(b, b)}"' for b, _n in m.continuing)
            prefix = f'"{seat}"' if seat is not None else "# seat ?"
            print(f"{prefix} = [{names}]  # {m.nickname}")

    # Saved for `draft preflight`, which runs offline: it can then name the
    # likely first-year keeps that are still shown as available.
    import datetime
    import json

    def player(b: str) -> dict:
        nhl = bare_to_nhl.get(b)
        return {
            "name": report.names.get(b, b),
            "nhl_id": nhl,
            "vorp": vorp.get(nhl) if nhl is not None else None,
            "adp": bare_adp.get(b),
        }

    saved = {
        "season": args.season,
        "league_key": key,
        "derived_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "from": report.seasons,
        "keeper_rounds": report.keeper_rounds,
        "declared": declared_rows,
        "managers": [
            {
                "nickname": m.nickname,
                "team_name": m.team_name,
                "team_key": m.current_team_key,
                "seat": seat_of.get(m.guid),
                "continuing": [{**player(b), "times_kept": n} for b, n in m.continuing],
                "expired": [player(b) for b in m.expired],
                "open_slots": m.open_slots(league.n_keepers),
                "candidates": sorted(
                    (player(b) for b in m.candidates),
                    key=lambda p: p["adp"] if p["adp"] is not None else 10_000,
                )[:10],
            }
            for m in report.managers
        ],
    }
    out = settings._resolve(Path("data")) / f"keepers-{args.season}.json"
    out.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    print()
    print(f"Saved for `ppilot draft preflight`: {out}")
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

    from puckpilot.draft.wsfeed import load_yahoo_names
    from puckpilot.yahoo.playermap import load_adp, resolve_adp_key

    adp, feed, ctx, pump_fn, league_key = None, None, None, None, None
    # The ADP source is named on its own, not inferred from whether --yahoo
    # happened to carry a dot: a bare --yahoo used to build the board on the
    # proxy ADP and say nothing. Replays use it too, so a rehearsal runs on
    # the same survival numbers the real draft will.
    if args.yahoo or args.adp_league_key or args.replay:
        league_key, notes = resolve_adp_key(conn, args.adp_league_key, args.yahoo)
        for note in notes:
            print(note, file=sys.stderr if note.startswith("WARNING") else sys.stdout)
        adp = load_adp(conn, league_key) if league_key else None
    if args.replay:
        # A draft that already happened, played back. Same poll(board) the
        # websocket drives, so this exercises the whole console offline - no
        # lobby, no browser, no forty minutes of waiting to find out the
        # interface is wrong.
        from puckpilot.draft.farm import load_all
        from puckpilot.draft.feed import ReplayFeed
        from puckpilot.draft.wsfeed import load_yahoo_id_map

        root = Path(args.replay)
        if root.is_file():
            # One named harvest. The old filter compared the path with itself,
            # so naming a file replayed whichever harvest sorted first.
            import json

            from puckpilot.draft.farm import MockResult

            harvests = [MockResult(**json.loads(root.read_text(encoding="utf-8")))]
        else:
            harvests = load_all(root)
        if not harvests:
            print(f"No harvested drafts under {root}", file=sys.stderr)
            return 2
        picks = harvests[0].picks
        feed = ReplayFeed(
            picks,
            load_yahoo_id_map(conn),
            interval=args.replay_interval,
            n_teams=harvests[0].n_teams,
            yahoo_names=load_yahoo_names(conn),
            drop=_parse_drop(args.replay_drop),
        )
        print(f"Replaying {len(picks)} picks at {args.replay_interval}s/pick.")
    elif args.yahoo:
        # The websocket carries picks in any Yahoo draft room, mock or real, and
        # is the only source measured at 100% on the picks it can map.
        from playwright.sync_api import sync_playwright

        from puckpilot.draft.wsfeed import WebsocketFeed, load_yahoo_id_map, pump

        profile = settings._resolve(Path("secrets/chrome-profile"))
        pw = sync_playwright().start()
        # This is the window the drafter drafts in: Yahoo allows one draft-room
        # connection per account, so a second browser would log this one out
        # and blind the feed. no_viewport lets the page follow the real window
        # instead of Playwright's fixed 1280x720, and it opens maximized.
        ctx = pw.chromium.launch_persistent_context(
            str(profile),
            headless=False,
            channel="chrome",
            no_viewport=True,
            args=["--start-maximized"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        with contextlib.suppress(Exception):
            page.goto(args.room, wait_until="domcontentloaded")
        import datetime

        frame_log = settings._resolve(Path("data/captures")) / (
            f"draft-frames-{datetime.datetime.now():%Y%m%d-%H%M%S}.log"
        )
        frame_log.parent.mkdir(parents=True, exist_ok=True)
        ws_feed = WebsocketFeed(
            ctx, load_yahoo_id_map(conn), load_yahoo_names(conn), frame_log=frame_log
        )
        feed = ws_feed
        results_key = args.yahoo if "." in str(args.yahoo) else league_key
        if results_key:
            # Second source: the room's own draft results, read through this
            # same logged-in browser from a background tab (not a draft-room
            # connection, so it cannot log the drafter out). On draft night
            # the websocket caught 2 of the first 6 picks; this lists them all,
            # and a restarted console catches up on every pick made so far.
            from puckpilot.draft.feed import CombinedFeed, YahooDraftFeed
            from puckpilot.yahoo.playermap import load_map
            from puckpilot.yahoo.session import YahooSession, YahooSessionError

            class _OpenPageSession(YahooSession):
                """Reads through whichever Yahoo page this browser already has
                open - never a tab of its own, which the drafter can close and
                which steals focus from the draft room when it opens."""

                def get(self, path: str) -> dict:
                    pages = [
                        p for p in ctx.pages if not p.is_closed() and "yahoo.com" in (p.url or "")
                    ]
                    if not pages:
                        raise YahooSessionError("no open Yahoo page to read draft results through")
                    self._page = pages[0]
                    return super().get(path)

            api = _OpenPageSession(profile, check_oauth=False)
            key_names = {
                str(k): str(n)
                for k, n in conn.execute("SELECT player_key, full_name FROM yahoo_player_map")
            }
            feed = CombinedFeed(
                [
                    ws_feed,
                    YahooDraftFeed(
                        api, results_key, load_map(conn, results_key), key_names, min_interval=3.0
                    ),
                ]
            )
            print(f"Draft results feed armed for {results_key} (every 3s).")
        print(f"Raw websocket frames -> {frame_log}")
        pump_fn = partial(pump, ctx)
        print(f"Websocket feed armed ({len(ws_feed.yahoo_to_nhl)} ids). Join your draft room.")

    board = build_live_board(
        conn,
        league,
        seat=args.seat,
        season=args.season,
        adp=adp,
        league_key=league_key,
        progress=print,
    )
    if args.web:
        return _serve_live(board, feed, ctx, args, league)
    run_live(board, feed=feed, cfg=LiveConfig(top=args.top), pump=pump_fn)
    return 0


def _cmd_draft_preflight(args: argparse.Namespace) -> int:
    """Everything that would silently corrupt the draft-night board, in one
    command that exits non-zero on any of it."""
    from pathlib import Path

    from puckpilot.data import store
    from puckpilot.preflight import run_preflight

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    # Load the file directly: the CLI's usual path falls back to generic
    # settings with one stderr line, which is exactly what this must catch.
    path = Path(args.league) if getattr(args, "league", None) else settings.resolved_league_path

    prober = None
    if not args.offline:
        from puckpilot.yahoo.probe import probe

        def prober():
            return probe(settings)

    import json

    saved = settings._resolve(Path("data")) / f"keepers-{args.season}.json"
    history = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else None

    report, _board = run_preflight(
        conn,
        path,
        seat=args.seat,
        season=args.season,
        adp_league_key=args.adp_league_key,
        prober=prober,
        progress=print if args.verbose else (lambda _m: None),
        keeper_history=history,
    )
    print(report.text)
    return 1 if report.failed else 0


def _parse_drop(raw: str | None) -> tuple[int, int] | None:
    """`--replay-drop 20:10` -> (20, 10): the feed goes dead after 20 room
    picks and misses the next 10, while the room keeps drafting."""
    if not raw:
        return None
    start, _, count = str(raw).partition(":")
    return int(start), int(count or 1)


def _parse_seats(raw: str | None, default: int) -> list[int]:
    """`--seats 4,7` -> [4, 7]. Empty means just the console's own seat."""
    if not raw:
        return [default]
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [default]


def _push_snapshots(url: str, key: str, state, seats: list[int]) -> str:
    """POST one snapshot per seat to the relay. Returns "" on success.

    Never raises: a relay that is down, redeploying, or unreachable must not
    take the local console with it. The drafter on this machine keeps working
    off the loopback view either way.
    """
    import urllib.error
    import urllib.request

    from puckpilot.web import wire

    try:
        # Inside the guard too. Building the snapshot or serializing it can fail
        # as well as the network can, and this runs in the console's main loop:
        # an exception here used to end the draft console, not just the push.
        body = wire.dumps({"seats": {str(s): state.snapshot(s) for s in seats}}).encode("utf-8")
    except Exception as e:
        return f"snapshot for push failed: {e.__class__.__name__}: {e}"
    try:
        # Constructing the request parses the URL, so a mistyped --publish
        # ("puckpilot-draft.azurecontainerapps.io", no scheme) raises here.
        req = urllib.request.Request(
            url.rstrip("/") + "/push",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-PuckPilot-Key": key},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
        return ""
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        return f"{e.__class__.__name__}: {e}"


def _serve_live(board, feed, ctx, args: argparse.Namespace, league) -> int:
    """Second-screen web view, pumped from the browser the feed is attached to.

    Three ways to reach it, in increasing order of exposure: loopback only (the
    default, and unchanged); `--share`, which binds every interface behind two
    generated tokens; and `--publish`, which additionally pushes each seat's
    snapshot to a relay so a second manager can watch from anywhere. The feed
    itself never moves - it is attached to the browser on this desk.
    """
    import os
    import webbrowser

    from puckpilot.draft.wsfeed import pump
    from puckpilot.web.access import Access
    from puckpilot.web.server import LiveState, serve

    state = LiveState(
        board=board,
        feed=feed,
        top=args.shortlist,
        board_rows=args.board_rows,
        cats=league.all_cats,
    )
    seats = _parse_seats(getattr(args, "seats", None), args.seat)
    for seat in seats:
        if not 0 <= seat < board.n_teams:
            print(f"seat {seat} outside 0..{board.n_teams - 1}", file=sys.stderr)
            return 2

    publish = getattr(args, "publish", None)
    push_key = getattr(args, "publish_key", None) or os.environ.get("PUCKPILOT_OWNER_KEY", "")
    if publish and not push_key:
        print(
            "--publish needs an owner key: pass --publish-key or set PUCKPILOT_OWNER_KEY.",
            file=sys.stderr,
        )
        return 2

    access = Access.generate() if getattr(args, "share", False) else None
    serve(state, port=args.port, access=access)

    local = f"http://127.0.0.1:{args.port}"
    print()
    if access is None:
        print(f"  Draft view: {local}")
    else:
        print(f"  Yours:  {local}/?seat={args.seat}&k={access.owner}")
        for seat in seats:
            if seat != args.seat:
                print(f"  Guest:  http://<this-machine>:{args.port}/?seat={seat}&k={access.guest}")
        print()
        print("  Guests can read any seat but cannot undo. Keys last for this run only.")
    if publish:
        print(f"  Relay:  {publish}  (seats {', '.join(str(s) for s in seats)})")
    print("  Ctrl+C to stop.")
    print()
    if not args.no_open:
        webbrowser.open(f"{local}/?seat={args.seat}" + (f"&k={access.owner}" if access else ""))

    # The web view must outlive anything the browser does - tabs opening and
    # closing, navigations, the draft room replacing the lobby. Nothing in here
    # is allowed to end the process except Ctrl+C.
    strikes = 0
    last_push = 0.0
    push_fails = 0
    try:
        while True:
            landed = 0
            try:
                landed = state.pump()
                if landed:
                    print(f"  +{landed} pick(s)  total={board.made}", flush=True)
                strikes = 0
            except Exception as e:
                strikes += 1
                if strikes in (1, 10, 50):
                    print(f"  feed poll failed ({e.__class__.__name__}); still serving", flush=True)
            if publish:
                # On change, plus a slow heartbeat so "seconds since last pick"
                # does not freeze on the relay between picks.
                now = time.time()
                if landed or now - last_push > 10:
                    err = _push_snapshots(publish, push_key, state, seats)
                    last_push = now
                    if err:
                        push_fails += 1
                        if push_fails in (1, 10, 100):
                            print(f"  relay push failed ({err}); still serving", flush=True)
                    else:
                        push_fails = 0
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
    say("This records what the room broadcasts. It never picks for you.")

    # Optional local helper: joining a lobby is a click, and the published tool
    # does not make it. If a user has written one it lives outside this repo.
    try:
        from puckpilot.local.join import join_a_mock
    except ImportError:
        join_a_mock = None

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
                if join_a_mock is None:
                    say("  join a mock draft in the browser; recording starts automatically")
                elif not join_a_mock(page, say):
                    say("  no room to join right now; moving on")
                    continue
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


def _cmd_draft_e2e(args: argparse.Namespace) -> int:
    """Whole drafts through board -> snapshot -> local page -> push -> relay -> guest."""
    import os
    from pathlib import Path

    import numpy as np

    from puckpilot.data import store
    from puckpilot.draft import e2e
    from puckpilot.draft.farm import load_all
    from puckpilot.draft.feed import ReplayFeed
    from puckpilot.draft.live import build_live_board
    from puckpilot.draft.wsfeed import WebsocketFeed, load_yahoo_id_map
    from puckpilot.web.relay import build_id
    from puckpilot.yahoo.playermap import load_adp

    def say(msg: str) -> None:
        print(msg, flush=True)

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    league = _league(args)
    quiet = (lambda _m: None) if not args.verbose else say

    if args.relay:
        owner = args.owner_key or os.environ.get("PUCKPILOT_E2E_OWNER_KEY", "")
        guest = args.guest_key or os.environ.get("PUCKPILOT_E2E_GUEST_KEY", "")
        if not owner or not guest:
            say("--relay needs both keys: --owner-key/--guest-key or PUCKPILOT_E2E_*_KEY")
            return 2
        relay = e2e.Relay(args.relay.rstrip("/"), owner, guest)
        say(f"relay: {relay.url}")
    else:
        relay = e2e.start_local_relay()
        say(f"relay: in-process at {relay.url}")

    keys = [r[0] for r in conn.execute("SELECT DISTINCT league_key FROM yahoo_player_map")]
    league_key = args.adp_league_key or (keys[0] if len(keys) == 1 else None)
    adp = load_adp(conn, league_key) if league_key else None
    say(f"ADP: {league_key or 'none'} ({len(adp or {})} players)")
    names = e2e.yahoo_rows(conn)
    idmap = load_yahoo_id_map(conn)
    seats = _parse_seats(args.seats, 0)
    expect = build_id() if args.expect_build else None
    sources = {"sim", "harvest", "capture"} if args.source == "all" else {args.source}
    results = []

    def run(board, polls, name, source, **kw):
        harness = e2e.Harness(
            board,
            relay,
            cats=kw.pop("cats"),
            seats=tuple(s for s in seats if s < board.n_teams),
            every=args.every,
            names=names,
            expect_build=expect,
            browser_every=args.browser,
            stale_check=args.stale_check,
        )
        try:
            res = harness.run(polls, name, source, **kw)
        except Exception as e:
            # One broken draft must not hide the rest of the run's results.
            res = e2e.E2EResult(name=name, source=source)
            res.violations.append(f"harness crashed: {e.__class__.__name__}: {e}")
        results.append(res)
        say(res.summary())

    try:
        if "sim" in sources:
            for i in range(args.drafts):
                seed = args.seed + i
                board = build_live_board(
                    conn, league, seat=seats[0], season=args.season, adp=adp,
                    league_key=league_key, progress=quiet,
                )  # fmt: skip
                rng = np.random.default_rng(seed)
                run(board, e2e.sim_polls(board, league, rng), f"sim seed {seed}", "sim",
                    cats=league.all_cats)  # fmt: skip

        if "harvest" in sources:
            rooms = load_all(Path(args.mocks))[: args.limit or None]
            for h in rooms:
                room = e2e.room_league(league, h.n_teams)
                board = build_live_board(
                    conn, room, seat=seats[0], season=args.season, adp=adp,
                    league_key=league_key, progress=quiet,
                )  # fmt: skip
                feed = ReplayFeed(h.picks, idmap, interval=0, n_teams=h.n_teams)
                polls = e2e.replay_polls(feed, len(h.picks) + 5)
                run(board, polls, f"room {h.started[:16]}", "harvest", cats=league.all_cats,
                    room_picks=len(h.picks), unmapped=lambda f=feed: f.unmapped,
                    feed=feed)  # fmt: skip

        if "capture" in sources:
            dirs = sorted(p.parent for p in Path(args.captures).glob("*/websocket.jsonl"))
            for d in dirs[: args.limit or None]:
                frames = e2e.capture_frames(d)
                feed = WebsocketFeed(e2e._NoContext(), idmap)
                for f in frames:  # size the room from the frames themselves
                    feed.ingest(f)
                n_teams = max((fr.seat for fr in feed.state.picks.values()), default=0)
                room_picks = len(feed.state.picks)
                if not n_teams:
                    say(f"SKIP  {d.name}: no pick frames")
                    continue
                feed = WebsocketFeed(e2e._NoContext(), idmap)
                room = e2e.room_league(league, n_teams)
                board = build_live_board(
                    conn, room, seat=seats[0], season=args.season, adp=adp,
                    league_key=league_key, progress=quiet,
                )  # fmt: skip
                run(board, e2e.frame_polls(frames, feed), f"capture {d.name}", "capture",
                    cats=league.all_cats, room_picks=room_picks,
                    unmapped=lambda f=feed: f.state.unmapped, feed=feed)  # fmt: skip
    finally:
        relay.close()

    failed = [r for r in results if not r.passed]
    picks = sum(r.picks for r in results)
    checks = sum(r.checks for r in results)
    say("")
    say(
        f"{len(results)} draft(s), {picks} picks, {checks} verified states, "
        f"{sum(r.pushes for r in results)} pushes: "
        + ("ALL PASS" if not failed else f"{len(failed)} FAILED")
    )
    return 1 if failed or not results else 0


def _season_setup(args, *, need_session: bool = False):
    """Manager, database, league key and (optionally) a Yahoo read session."""
    from puckpilot.season import cli_support

    manager = cli_support.resolve_manager(getattr(args, "manager", None))
    conn = cli_support.open_db(manager)
    league_key = cli_support.resolve_league_key(conn, manager)
    session = cli_support.open_session(manager) if need_session else None
    return manager, conn, league_key, session


def _cmd_season_settings(args: argparse.Namespace) -> int:
    from puckpilot.season import cli_support
    from puckpilot.season.fetch import fetch_runtime, load_runtime, save_runtime
    from puckpilot.season.manager import ManagerError

    try:
        manager, conn, league_key, _ = _season_setup(args)
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    runtime = None if args.refresh else load_runtime(conn, league_key)
    if runtime is None:
        try:
            runtime = cli_support.run_session(
                manager,
                lambda s: fetch_runtime(s, league_key, manager.team_key, progress=print),
            )
        except (ManagerError, cli_support.SeasonCliError) as e:
            return cli_support.report(e)
        save_runtime(conn, runtime)
        print(f"saved settings for {league_key}")

    print()
    print(
        f"{runtime.name}  ({runtime.league_key})  {runtime.num_teams} teams, {runtime.scoring_type}"
    )
    print(f"  season      {runtime.start_date} -> {runtime.end_date}  ({runtime.nhl_season})")
    print(
        f"  weeks       {runtime.start_week}-{runtime.end_week}, "
        f"{runtime.regular_weeks} regular, playoffs from {runtime.playoff_start_week} "
        f"({runtime.num_playoff_teams} teams)"
    )
    print(
        f"  calendar    {len(runtime.weeks)} week(s) fetched; current week {runtime.current_week}"
    )
    mode = "daily" if runtime.is_daily_lineup else "weekly"
    print(f"  lineups     {mode} ({runtime.weekly_deadline})")
    print(
        f"  waivers     type {runtime.waiver_type}, {runtime.waiver_days}-day, "
        f"FAAB {'yes' if runtime.uses_faab else 'no'}"
    )
    print(
        f"  adds        {runtime.max_weekly_adds} per week, {runtime.max_adds} per season; "
        f"min games {runtime.min_games_played}"
    )
    print(f"  slots       {', '.join(f'{s.position}x{s.count}' for s in runtime.slots)}")
    print(f"  fetched     {runtime.fetched_at}")
    return 0


def _cmd_season_roster(args: argparse.Namespace) -> int:
    from puckpilot.season import cli_support
    from puckpilot.season.fetch import discover_team_key, fetch_roster, save_roster
    from puckpilot.season.manager import ManagerError
    from puckpilot.yahoo import playermap

    try:
        manager, conn, league_key, _ = _season_setup(args)
        pmap = playermap.load_map(conn, league_key)

        def _read(session):
            key = manager.team_key or discover_team_key(session, league_key)
            return fetch_roster(session, key, args.date, player_map=pmap)

        roster = cli_support.run_session(manager, _read)
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    save_roster(conn, manager.name, roster)
    print(f"{roster.team_key}  {roster.date}  {len(roster)} players")
    print()
    for p in roster.players:
        flag = f"  [{p.status_full or p.status}{' - ' + p.injury_note if p.injury_note else ''}]"
        print(
            f"  {p.selected_slot:5} {p.name:24} {p.team:4} "
            f"{'/'.join(sorted(p.eligible)):8}{flag if p.status else ''}"
        )
    if roster.unmapped:
        print()
        print(f"  not in the player map: {', '.join(roster.unmapped)}")
        print("  run `ppilot yahoo playermap --reresolve-only` after a data sync")
    return 0


def _cmd_lineup_today(args: argparse.Namespace) -> int:
    from puckpilot.season import cli_support
    from puckpilot.season.fetch import discover_team_key, fetch_roster, save_roster
    from puckpilot.season.goalies import ChainedGoalieSource, TrailingStartShareSource
    from puckpilot.season.manager import ManagerError
    from puckpilot.season.today import build_plan, yahoo_goalie_games
    from puckpilot.season.values import build_value_model
    from puckpilot.yahoo import playermap

    date = args.date or cli_support.today_str()
    try:
        manager, conn, league_key, _ = _season_setup(args)
        runtime = cli_support.load_rules(conn, league_key)
        pmap = playermap.load_map(conn, league_key)

        def _read(session):
            key = manager.team_key or discover_team_key(session, league_key)
            return fetch_roster(session, key, date, player_map=pmap)

        roster = cli_support.run_session(manager, _read)
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    save_roster(conn, manager.name, roster)
    season = runtime.nhl_season
    train = [s for s in DEFAULT_SEASONS if s < season][-3:][::-1]
    values = build_value_model(conn, season, tuple(train), manager.league)
    goalies = ChainedGoalieSource(TrailingStartShareSource(conn, season, fallback_season=train[0]))

    from puckpilot.season import protocol as protocol_mod

    weights: dict[str, float] = {}
    if manager.authority.lineup.follow_protocol:
        try:
            live = protocol_mod.active(conn, manager.name, league_key, runtime.week_of(date))
            weights = live.weights() if live else {}
        except Exception:  # noqa: BLE001 - no calendar, no protocol; the plain plan stands
            weights = {}

    plan = build_plan(
        conn,
        runtime,
        roster,
        values,
        goalies,
        date,
        manager=manager.name,
        authority=manager.authority.lineup,
        goalie_starts_so_far=yahoo_goalie_games(roster, runtime, date),
        weights=weights,
    )
    from puckpilot.season import explain

    reasons = explain.move_reasons(conn, runtime, plan)
    print()
    for line in explain.plan_story(conn, runtime, plan):
        print(f"  {line}" if line else "")
    if plan.moves:
        print()
        print("  What to change in Yahoo:")
        for m in plan.moves:
            print(f"    {m.describe()}")
            why = reasons.get(m.player.player_key)
            if why:
                print(f"        {why}")
        print()
        print(f"  Worth {plan.gain:+.2f} for tonight, in the same units the board used.")
    print()
    print(f"  {plan.authority_reason}")
    _season_publish(manager, conn, league_key, plan=plan, roster=roster, reasons=reasons)
    if args.explain:
        print()
        print("  value tonight:")
        for c in sorted(plan.playing, key=lambda c: -c.value):
            extra = f"  {c.note}" if c.note else ""
            p = f"  P(start) {c.p_start:.0%}" if c.p_start is not None else ""
            print(f"    {c.value:+7.2f}  {c.player.name:24}{p}{extra}")
    return 0


def _season_publish(
    manager, conn, league_key, plan=None, week_plan=None, roster=None, reasons=None, quiet=False
):
    """Collect decisions, then push the view. Never fatal."""
    from puckpilot.season import publish, snapshot

    if not manager.page.publishes:
        return
    key = manager.page.owner_key or os.environ.get("PUCKPILOT_MANAGER_KEY", "")
    if not key:
        if not quiet:
            print("  (page configured but no key - set PUCKPILOT_MANAGER_KEY)")
        return
    try:
        decisions = publish.collect(manager.page.url, key)
        for line in snapshot.apply_decisions(conn, decisions):
            print(f"  from your phone: {line}")
        named = roster or week_plan
        snap = snapshot.build(
            conn,
            manager.name,
            league_key,
            team_name=getattr(named, "team_name", "") or manager.name,
            plan=plan,
            week_plan=week_plan,
            roster=roster,
            reasons=reasons,
        )
        publish.push(manager.page.url, key, snap)
        if not quiet:
            print(f"  published to {manager.page.url}")
    except publish.PublishError as e:
        print(f"  page not updated: {e}")


def _cmd_season_locks(args: argparse.Namespace) -> int:
    """When today's slots actually close, and by when a run has to happen."""
    from puckpilot.season import cli_support, locks
    from puckpilot.season.manager import ManagerError

    try:
        manager, conn, league_key, _ = _season_setup(args)
        runtime = cli_support.load_rules(conn, league_key)
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    day = args.date or cli_support.today_str()
    teams = locks.roster_teams(conn, manager.name)
    if not teams:
        print("No roster snapshot yet - run `ppilot season roster` first.")
        return 2
    todays = locks.locks_for(conn, runtime.nhl_season, day, teams)
    print(locks.describe(day, todays))
    ahead = locks.upcoming(todays)
    if day == cli_support.today_str():
        print()
        if ahead:
            print("  Still to come: " + ", ".join(x.pretty for x in ahead))
            print("  Runs would be scheduled at: " + ", ".join(locks.run_times(ahead)))
        else:
            print("  Everything of yours has locked for today.")
    return 0


def _cmd_season_schedule(args: argparse.Namespace) -> int:
    """Register (or show, or remove) the daily runs."""
    import os

    from puckpilot.config import REPO_ROOT
    from puckpilot.season import cli_support, schedule
    from puckpilot.season.manager import ManagerError

    try:
        manager = cli_support.resolve_manager(getattr(args, "manager", None))
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    times = tuple(t.strip() for t in args.at.split(",") if t.strip())
    items = schedule.tasks(manager.name, REPO_ROOT, times)

    already = schedule.installed()
    if already:
        print("Already registered: " + ", ".join(already))
        print()

    if args.remove:
        for line in schedule.remove(manager.name):
            print(line)
        return 0

    print(schedule.describe(items, bool(os.environ.get("PUCKPILOT_MANAGER_KEY"))))
    if not args.install:
        print()
        print("Nothing registered. Add --install to set these up, --remove to take them away.")
        return 0

    print()
    for line in schedule.install(items):
        print(line)
    print()
    print("Remove them with: ppilot season schedule --remove")
    return 0


def _cmd_season_run(args: argparse.Namespace) -> int:
    """Everything a day needs, in order. Built to be scheduled."""
    from puckpilot.season import cli_support
    from puckpilot.season.manager import ManagerError
    from puckpilot.season.run import run_day

    try:
        manager, conn, league_key, _ = _season_setup(args)
        runtime = cli_support.load_rules(conn, league_key)
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    # Today, not `runtime.current_date` - that is a cached field, so a job
    # running on a settings snapshot from last week would quietly plan last
    # week's day and look entirely healthy doing it.
    day = args.date or cli_support.today_str()
    lines: list[str] = []

    def say(text):
        lines.append(text)
        print(text)

    report = run_day(
        conn,
        manager,
        league_key,
        runtime,
        day,
        weekly=True if args.weekly else (False if args.no_weekly else None),
        do_sync=not args.no_sync,
        reschedule=not args.no_reschedule,
        say=say,
    )
    if args.log:
        path = Settings()._resolve(Path(args.log))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(chr(10).join(lines) + chr(10) * 2)
    return 1 if report.failed else 0


def _cmd_season_gate(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.season.replay import protocol_gate_report

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    report = protocol_gate_report(
        conn,
        season=args.season,
        n_teams_tested=args.teams,
        seed=args.seed,
        league=_league(args),
        progress=print if args.verbose else None,
    )
    print()
    print(report.text)
    return 0


def _cmd_season_preflight(args: argparse.Namespace) -> int:
    from puckpilot.season import cli_support
    from puckpilot.season import preflight as pf
    from puckpilot.season.fetch import discover_team_key, fetch_roster, load_runtime
    from puckpilot.season.goalies import ChainedGoalieSource, TrailingStartShareSource
    from puckpilot.season.manager import ManagerError
    from puckpilot.season.today import build_plan, yahoo_goalie_games
    from puckpilot.season.values import build_value_model
    from puckpilot.yahoo import playermap

    report = pf.SeasonPreflightReport()
    try:
        manager, conn, league_key, _ = _season_setup(args)
    except (ManagerError, cli_support.SeasonCliError) as e:
        print(f"puckpilot: {e}")
        return 2

    report.checks.append(pf.check_manager(manager))
    report.checks.append(pf.check_authority(manager))

    runtime = load_runtime(conn, league_key)
    report.checks.append(pf.check_runtime(runtime))
    report.checks.append(pf.check_categories(runtime, manager.league))
    day = args.date or cli_support.today_str()
    report.checks.append(pf.check_calendar(runtime, day))
    report.checks.append(pf.check_player_map(conn, league_key))

    roster = None
    if runtime is not None:
        season = runtime.nhl_season
        report.checks.append(pf.check_schedule(conn, season, day))
        report.checks.append(pf.check_data_freshness(conn, season, day))
        if not args.offline:
            pmap = playermap.load_map(conn, league_key)

            def _read(session):
                key = manager.team_key or discover_team_key(session, league_key)
                return fetch_roster(session, key, day, player_map=pmap)

            try:
                roster = cli_support.run_session(manager, _read)
            except (ManagerError, cli_support.SeasonCliError) as e:
                report.checks.append(pf.Check("yahoo session", pf.FAIL, str(e)))
        report.checks.append(
            pf.check_roster(roster)
            if roster or not args.offline
            else pf.Check("roster", pf.INFO, "skipped (--offline)")
        )

        train = [x for x in DEFAULT_SEASONS if x < season][-3:][::-1]
        values = build_value_model(conn, season, tuple(train), manager.league)
        report.checks.append(pf.check_projections(values, roster))
        goalies = ChainedGoalieSource(
            TrailingStartShareSource(conn, season, fallback_season=train[0])
        )
        report.checks.append(pf.check_goalies(goalies, roster, day))

        if roster is not None:
            plan = build_plan(
                conn,
                runtime,
                roster,
                values,
                goalies,
                day,
                manager=manager.name,
                authority=manager.authority.lineup,
                goalie_starts_so_far=yahoo_goalie_games(roster, runtime, day),
            )
            report.checks.append(pf.check_plan(plan))

    page_key = manager.page.owner_key or os.environ.get("PUCKPILOT_MANAGER_KEY", "")
    report.checks.append(pf.check_page(manager, page_key))
    print(report.text)
    return 1 if report.failed else 0


def _cmd_season_protocol(args: argparse.Namespace) -> int:
    from puckpilot.season import cli_support
    from puckpilot.season import protocol as protocol_mod
    from puckpilot.season.manager import ManagerError

    try:
        manager, conn, league_key, _ = _season_setup(args)
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    try:
        for pid in args.approve or []:
            print(protocol_mod.decide(conn, pid, True).describe())
        for pid in args.reject or []:
            print(protocol_mod.decide(conn, pid, False).describe())
    except protocol_mod.ProtocolError as e:
        print(f"puckpilot: {e}")
        return 2
    if args.approve or args.reject:
        return 0

    from puckpilot.season import explain

    rows = protocol_mod.listing(conn, manager.name, league_key)
    if not rows:
        print("No protocols yet - run `ppilot season week` to propose one.")
        return 0
    for p in rows:
        print(f"[{p.status.upper()}] #{p.id}")
        for line in explain.protocol_story(p):
            print(f"  {line}" if line else "")
        print()
    return 0


def _cmd_season_proposals(args: argparse.Namespace) -> int:
    from puckpilot.season import cli_support, proposals
    from puckpilot.season.manager import ManagerError

    try:
        manager, conn, league_key, _ = _season_setup(args)
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    try:
        for pid in args.approve or []:
            print(proposals.decide(conn, pid, True).describe())
        for pid in args.reject or []:
            print(proposals.decide(conn, pid, False).describe())
    except proposals.ProposalError as e:
        print(f"puckpilot: {e}")
        return 2
    if args.approve or args.reject:
        return 0

    rows = proposals.listing(conn, manager.name, league_key, status=args.status)
    if not rows:
        print("No proposals.")
        return 0
    for p in rows:
        print(p.describe())
        if p.reason.get("timing"):
            print(f"      {p.reason['timing']}")
    print()
    print("  Approve with: ppilot season proposals --approve <id>")
    return 0


def _cmd_season_week(args: argparse.Namespace) -> int:
    from puckpilot.draft.sim import build_universe
    from puckpilot.season import cli_support, pool
    from puckpilot.season import week as weekmod
    from puckpilot.season.fetch import fetch_matchups, fetch_roster, save_pool_if_any
    from puckpilot.season.goalies import ChainedGoalieSource, TrailingStartShareSource
    from puckpilot.season.manager import ManagerError
    from puckpilot.season.matchups import current_or_next, for_week
    from puckpilot.season.values import build_value_model
    from puckpilot.yahoo import playermap

    try:
        manager, conn, league_key, _ = _season_setup(args)
        runtime = cli_support.load_rules(conn, league_key)
        pmap = playermap.load_map(conn, league_key)
        today = args.date or cli_support.today_str()

        def _read(session):
            team_key = manager.team_key or runtime.league_key
            matchups = fetch_matchups(session, team_key)
            m = for_week(matchups, args.week) if args.week else current_or_next(matchups, today)
            if m is None:
                m = for_week(matchups, runtime.current_week)
            if m is None:
                raise cli_support.SeasonCliError(
                    f"no matchup covering {today} and none after it; the season is over."
                )
            ours = fetch_roster(session, team_key, m.start, player_map=pmap)
            theirs = (
                fetch_roster(session, m.opponent_key, m.start, player_map=pmap)
                if m.opponent_key
                else ours
            )
            fa = pool.fetch_pool(
                session, league_key, "FA", limit=args.pool, player_map=pmap, progress=print
            )
            wv = pool.fetch_pool(session, league_key, "W", limit=50, player_map=pmap)
            return m, ours, theirs, fa + wv

        m, ours, theirs, available = cli_support.run_session(manager, _read)
    except (ManagerError, cli_support.SeasonCliError) as e:
        return cli_support.report(e)

    save_pool_if_any(conn, league_key, m.start, available)
    season = runtime.nhl_season
    train = [s for s in DEFAULT_SEASONS if s < season][-3:][::-1]
    universe = build_universe(conn, season, tuple(train), manager.league)
    values = build_value_model(conn, season, tuple(train), manager.league)
    goalies = ChainedGoalieSource(TrailingStartShareSource(conn, season, fallback_season=train[0]))

    plan = weekmod.build_week_plan(
        conn,
        runtime,
        manager.league,
        m.as_week(),
        m.opponent_name,
        ours,
        theirs,
        available,
        universe.frame,
        goalies,
        values,
        min_gain=manager.authority.transactions.min_weekly_gain,
        max_targets=args.top,
    )
    from puckpilot.season import explain

    print()
    for line in explain.week_story(plan, runtime):
        print(f"  {line}" if line else "")
    print()
    print(plan.text())
    print()
    stuck = ours.illegal_ir()
    if args.propose and plan.targets and stuck:
        print(
            "  Not queued: " + ", ".join(p.name for p in stuck) + " must come off IR first -"
            " Yahoo refuses every add and drop until then."
        )
    elif args.propose and plan.targets:
        from puckpilot.season import proposals

        made = proposals.propose(
            conn,
            manager.name,
            league_key,
            ours.team_key,
            plan.targets,
            plan.week,
            max_pending=manager.authority.transactions.max_pending,
        )
        if made:
            print(f"  Queued {len(made)} proposal(s) for your decision:")
            for pr in made:
                print(f"    {pr.describe()}")
            print("  Decide with: ppilot season proposals --approve N  (or --reject N)")
        else:
            print("  Nothing new to queue - these are already awaiting your decision.")
    else:
        print("  Nothing here is executed. Approve a move to act on it.")

    from puckpilot.season import protocol as protocol_mod

    stance = protocol_mod.derive(
        plan.outlook, manager.name, league_key, ours.team_key, plan.week, m.opponent_name
    )
    existing = protocol_mod.load(conn, manager.name, league_key, plan.week)
    live = existing if existing and existing.status == protocol_mod.APPROVED else None
    if live is None:
        stance = protocol_mod.save(conn, stance)
    shown = live or stance
    print()
    for line in explain.protocol_story(shown, plan.adds_left_week):
        print(f"  {line}" if line else "")
    print()
    if live is not None:
        print(f"  Already approved (protocol #{live.id}).")
    else:
        print(f"  Approve with: ppilot season protocol --approve {shown.id}")
        print("  Or just ignore it - nothing happens until you decide.")

    _season_publish(manager, conn, league_key, week_plan=plan, roster=ours)

    if args.trending:
        hot = pool.rising(available, limit=10)
        print()
        if not hot:
            print("  Nobody is being picked up yet - Yahoo's delta is week over week.")
        else:
            print("  The league is picking these up:")
        for x in hot:
            print(
                f"    +{x.percent_owned_delta:>4.0f}%  {x.name:22} "
                f"{x.team:4} ({x.percent_owned:.0f}% owned)"
            )
    return 0


def _cmd_lineup_verify(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.season.replay import live_policy_report

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    report = live_policy_report(
        conn,
        season=args.season,
        n_drafts=args.drafts,
        seed=args.seed,
        league=_league(args),
        min_gain=args.min_gain,
        progress=print,
    )
    print()
    print(report.text)
    # The live path exists to run the validated policy, not a different one.
    live = report.totals["live_perfect"]
    validated = report.totals["validated_perfect"]
    if validated and (live - validated) / validated < -0.02:
        print()
        print("FAIL: the live path is more than 2% behind the policy it replaces.")
        return 1
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


def _cmd_data_daily(args: argparse.Namespace) -> int:
    from puckpilot.data import store
    from puckpilot.data.nhl import NhlClient
    from puckpilot.data.sync import sync_day

    settings = Settings()
    conn = store.connect(settings.resolved_db_path)
    store.init_db(conn)
    nhl = NhlClient()
    out = sync_day(conn, nhl, args.season, max_players=args.max_players, progress=print)
    print(
        f"{args.season}: {out['boxscores']} boxscore(s), "
        f"{out['players_synced']} player log(s), {out['still_behind']} still behind"
    )
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


def schedule_times() -> tuple[str, ...]:
    from puckpilot.season.schedule import DEFAULT_TIMES

    return DEFAULT_TIMES


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
    yl.add_argument("--league-key", default=None, help="e.g. 465.l.12345 (default: auto-detect)")
    yl.add_argument("--picks", type=int, default=10, help="Draft picks to print")
    yl.set_defaults(func=_cmd_yahoo_league)

    yw = yahoo_sub.add_parser(
        "watch-draft",
        help="Poll draftresults through a live draft to prove whether it updates live",
    )
    yw.add_argument("--league-key", default=None, help="e.g. 465.l.12345 (default: auto-detect)")
    yw.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds between polls (default: watch.DEFAULT_INTERVAL_S)",
    )
    yw.add_argument("--duration", type=float, default=5400.0, help="Give up after N seconds")
    yw.set_defaults(func=_cmd_yahoo_watch)

    ym = yahoo_sub.add_parser(
        "playermap",
        help="Build the Yahoo player-key -> NHL id map and Yahoo's own ADP",
    )
    ym.add_argument("--league-key", default=None, help="e.g. 465.l.12345 (default: auto-detect)")
    ym.add_argument("--limit", type=int, default=600, help="How deep into Yahoo's pool to fetch")
    ym.add_argument(
        "--reresolve-only",
        action="store_true",
        help="Skip Yahoo; re-match already-fetched unmatched rows against nhl_players now",
    )
    ym.set_defaults(func=_cmd_yahoo_playermap)

    yk = yahoo_sub.add_parser(
        "keepers",
        help="Reconstruct keeper contracts from the league's Yahoo draft history (prints only)",
    )
    yk.add_argument("--league-key", default=None, help="e.g. 477.l.12345 (default: auto-detect)")
    yk.add_argument("--season", default="20262027", help="Season the keepers are for")
    yk.add_argument(
        "--order",
        default=None,
        metavar="A,B,...",
        help="Draft order as manager nicknames or team keys, first pick first - used for "
        "seat numbers until Yahoo sets draft positions itself",
    )
    yk.add_argument(
        "--candidates", type=int, default=4, help="First-year keep candidates to list per team"
    )
    yk.set_defaults(func=_cmd_yahoo_keepers)

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

    daily = data_sub.add_parser(
        "daily", help="In-season catch-up: last night's boxscores and the logs behind them"
    )
    daily.add_argument("--season", default="20262027")
    daily.add_argument(
        "--max-players", type=int, default=None, help="Cap player log fetches for a quick run"
    )
    daily.set_defaults(func=_cmd_data_daily)

    rosters = data_sub.add_parser(
        "rosters",
        help="Repoint player teams at the current season's NHL rosters",
    )
    rosters.add_argument(
        "--season",
        default="20262027",
        help="Season whose rosters are authoritative (default 20262027)",
    )
    rosters.set_defaults(func=_cmd_data_rosters)

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

    e2e = draft_sub.add_parser(
        "e2e",
        help="Whole drafts through board, page, push and relay, verified at every pick "
        "(bots, harvested rooms, recorded websocket frames - never a live room)",
    )
    e2e.add_argument("--source", choices=("all", "sim", "harvest", "capture"), default="all")
    e2e.add_argument("--drafts", type=int, default=3, help="Bot drafts on the real league")
    e2e.add_argument("--seed", type=int, default=20260918)
    e2e.add_argument("--seats", default="0,5", metavar="N,N", help="Seats to verify and push")
    e2e.add_argument("--every", type=int, default=1, help="Verify every N picks")
    e2e.add_argument("--season", default="20262027")
    e2e.add_argument("--limit", type=int, default=0, help="Max rooms per replay source")
    e2e.add_argument("--mocks", default="data/mocks")
    e2e.add_argument("--captures", default="data/captures")
    e2e.add_argument("--adp-league-key", default=None, help="Default: the only key in the map")
    e2e.add_argument(
        "--relay", default=None, metavar="URL", help="Remote relay (default: in-process)"
    )
    e2e.add_argument("--owner-key", default=None, help="Default: PUCKPILOT_E2E_OWNER_KEY")
    e2e.add_argument("--guest-key", default=None, help="Default: PUCKPILOT_E2E_GUEST_KEY")
    e2e.add_argument(
        "--expect-build",
        action="store_true",
        help="Fail unless the relay's /healthz build matches this checkout",
    )
    e2e.add_argument(
        "--browser",
        type=int,
        default=0,
        metavar="N",
        help="Also open the guest link in headless Chrome (temporary profile) and check "
        "the rendered page every N picks",
    )
    e2e.add_argument(
        "--stale-check",
        action="store_true",
        help="With --browser: after the draft, stop pushing and require NOT LIVE",
    )
    e2e.add_argument("--verbose", action="store_true", help="Show board-building output")
    e2e.set_defaults(func=_cmd_draft_e2e)

    farm = draft_sub.add_parser(
        "farm",
        help="Sit through Yahoo mock drafts and harvest pool-coverage / survival data",
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
        help="How long to wait for a draft to start before giving up on a room "
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

    pre = draft_sub.add_parser(
        "preflight",
        help="Check everything that would silently corrupt the draft-night board; "
        "exits non-zero on any FAIL",
    )
    pre.add_argument("--seat", type=int, required=True, help="Your draft slot (0-based)")
    pre.add_argument("--season", default="20262027", help="Season being drafted")
    pre.add_argument(
        "--adp-league-key",
        default=None,
        metavar="KEY",
        help="League key whose Yahoo ADP the board uses (default: the only mapped league)",
    )
    pre.add_argument(
        "--offline", action="store_true", help="Skip the Yahoo OAuth probe (the only network call)"
    )
    pre.add_argument("--verbose", action="store_true", help="Also print the board build log")
    pre.set_defaults(func=_cmd_draft_preflight)

    live = draft_sub.add_parser("live", help="Draft-night console: live recommendations")
    live.add_argument("--seat", type=int, default=0, help="Your draft slot (0-based)")
    live.add_argument("--season", default="20262027", help="Season to draft for")
    live.add_argument(
        "--top", type=int, default=12, help="Candidates in the terminal console's table"
    )
    live.add_argument(
        "--shortlist",
        type=int,
        default=3,
        help="Reasoned cards in the web view (default 3 - the point is to be readable "
        "on a 30-second clock, not exhaustive)",
    )
    live.add_argument(
        "--board-rows", type=int, default=300, help="Rows in the web view's remaining board"
    )
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
        "--adp-league-key",
        default=None,
        metavar="KEY",
        help="League key whose Yahoo ADP to use (default: the key given to --yahoo, "
        "else the only league in the player map). Without ADP the board runs on a "
        "proxy, and says so.",
    )
    live.add_argument(
        "--room",
        default="https://hockey.fantasysports.yahoo.com/hockey",
        help="Page to open when --yahoo is used",
    )
    live.add_argument(
        "--replay",
        default=None,
        metavar="PATH",
        help="Play back a harvested draft (data/mocks) instead of reading a live room - "
        "exercises the whole console offline",
    )
    live.add_argument(
        "--replay-interval",
        type=float,
        default=0.35,
        help="Seconds per pick when replaying (0 = as fast as possible)",
    )
    live.add_argument(
        "--replay-drop",
        default=None,
        metavar="START:COUNT",
        help="Failure drill: the replayed feed goes silent after START room picks and "
        "misses COUNT of them while the room keeps drafting. Recover by hand.",
    )
    live.add_argument(
        "--web",
        action="store_true",
        help="Serve the second-screen view (shortlist + full board + roster) instead "
        "of the terminal console",
    )
    live.add_argument("--port", type=int, default=8765, help="Port for --web")
    live.add_argument(
        "--share",
        action="store_true",
        help="Bind every interface behind generated owner/guest keys, so a second "
        "manager in the same draft can open the view. Guests read any seat but "
        "cannot undo. Without this the view stays loopback-only.",
    )
    live.add_argument(
        "--seats",
        default=None,
        metavar="N,N",
        help="Seats to serve views for (default: just --seat). Everyone in the room "
        "shares one board; only the roster and shortlist differ by seat.",
    )
    live.add_argument(
        "--publish",
        default=None,
        metavar="URL",
        help="Also push each seat's snapshot to a relay at this URL, so the view "
        "is reachable from anywhere. The feed stays on this machine.",
    )
    live.add_argument(
        "--publish-key",
        default=None,
        help="Owner key for --publish (default: PUCKPILOT_OWNER_KEY)",
    )
    live.add_argument("--no-open", action="store_true", help="Do not open a browser tab")
    live.add_argument(
        "--interval", type=float, default=1.0, help="Seconds between feed polls with --web"
    )
    live.set_defaults(func=_cmd_draft_live)

    # In-season commands all answer "for whom, and on what day", so they share
    # a parent parser rather than redeclaring the flags five times.
    seasonal = argparse.ArgumentParser(add_help=False)
    seasonal.add_argument(
        "--manager",
        default=None,
        metavar="NAME",
        help="Manager config in managers/ (optional when only one exists)",
    )
    seasonal.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD", help="Day to act on (default: today)"
    )

    season = sub.add_parser("season", help="In-season league state")
    season_sub = season.add_subparsers(dest="subcommand", required=True)
    s_settings = season_sub.add_parser(
        "settings", parents=[seasonal], help="The league's own rules, read from Yahoo"
    )
    s_settings.add_argument(
        "--refresh", action="store_true", help="Re-read from Yahoo instead of the cache"
    )
    s_settings.set_defaults(func=_cmd_season_settings)
    s_roster = season_sub.add_parser(
        "roster", parents=[seasonal], help="Your current roster, with slots and injuries"
    )
    s_roster.set_defaults(func=_cmd_season_roster)

    s_pre = season_sub.add_parser(
        "preflight", parents=[seasonal], help="Is everything current enough to act on?"
    )
    s_pre.add_argument(
        "--offline", action="store_true", help="Skip the Yahoo reads and check the rest"
    )
    s_pre.set_defaults(func=_cmd_season_preflight)

    s_gate = season_sub.add_parser(
        "gate", help="Does a week protocol win more categories? (scored on categories, not value)"
    )
    s_gate.add_argument("--season", default="20252026")
    s_gate.add_argument("--teams", type=int, default=6)
    s_gate.add_argument("--seed", type=int, default=123)
    s_gate.add_argument("--verbose", action="store_true")
    s_gate.set_defaults(func=_cmd_season_gate)

    s_run = season_sub.add_parser(
        "run",
        parents=[seasonal],
        help="Everything for today in one go: sync, decisions, lineup, page (and Mondays)",
    )
    s_run.add_argument(
        "--weekly", action="store_true", help="Also do the weekly plan, whatever day it is"
    )
    s_run.add_argument(
        "--no-weekly", action="store_true", help="Skip the weekly plan even on week one's first day"
    )
    s_run.add_argument("--no-sync", action="store_true", help="Skip the data catch-up")
    s_run.add_argument(
        "--no-reschedule",
        action="store_true",
        help="Do not re-plan today's lock-timed runs",
    )
    s_run.add_argument(
        "--log", default="data/logs/season.log", metavar="PATH", help="Append the run to this file"
    )
    s_run.set_defaults(func=_cmd_season_run)

    s_sched = season_sub.add_parser(
        "schedule", parents=[seasonal], help="Run it daily without being asked (Task Scheduler)"
    )
    s_sched.add_argument("--install", action="store_true", help="Actually register the tasks")
    s_sched.add_argument("--remove", action="store_true", help="Take them away again")
    s_sched.add_argument(
        "--at",
        default=",".join(schedule_times()),
        metavar="HH:MM,...",
        help="Local times to run at",
    )
    s_sched.set_defaults(func=_cmd_season_schedule)

    s_locks = season_sub.add_parser(
        "locks", parents=[seasonal], help="When today's slots close, player by player"
    )
    s_locks.set_defaults(func=_cmd_season_locks)

    s_week = season_sub.add_parser(
        "week", parents=[seasonal], help="This week's category plan and add targets"
    )
    s_week.add_argument("--week", type=int, default=None, help="Fantasy week (default: today's)")
    s_week.add_argument("--pool", type=int, default=150, help="Free agents to consider")
    s_week.add_argument("--top", type=int, default=5, help="Targets to propose")
    s_week.add_argument("--trending", action="store_true", help="Also show rising ownership")
    s_week.add_argument(
        "--propose", action="store_true", help="Queue the targets for your approval"
    )
    s_week.set_defaults(func=_cmd_season_week)

    s_prop = season_sub.add_parser(
        "proposals", parents=[seasonal], help="Review, approve or reject queued transactions"
    )
    s_prop.add_argument("--status", default=None, help="pending / approved / rejected / executed")
    s_prop.add_argument("--approve", type=int, nargs="*", metavar="ID")
    s_prop.add_argument("--reject", type=int, nargs="*", metavar="ID")
    s_prop.set_defaults(func=_cmd_season_proposals)

    s_proto = season_sub.add_parser(
        "protocol",
        parents=[seasonal],
        help="The week's agreed category stance: what to chase and what to give up",
    )
    s_proto.add_argument("--approve", type=int, nargs="*", metavar="ID")
    s_proto.add_argument("--reject", type=int, nargs="*", metavar="ID")
    s_proto.set_defaults(func=_cmd_season_protocol)

    lineup = sub.add_parser("lineup", help="Daily lineup tools")
    lineup_sub = lineup.add_subparsers(dest="subcommand", required=True)
    today = lineup_sub.add_parser(
        "today", parents=[seasonal], help="Tonight's lineup changes, as a diff"
    )
    today.add_argument(
        "--explain", action="store_true", help="Show each playing player's value tonight"
    )
    today.set_defaults(func=_cmd_lineup_today)
    verify = lineup_sub.add_parser(
        "verify", help="Score the live daily path against the validated optimizer"
    )
    verify.add_argument("--season", default="20252026", help="Season to replay")
    verify.add_argument("--drafts", type=int, default=1, help="Drafts to source rosters from")
    verify.add_argument("--seed", type=int, default=123)
    verify.add_argument("--min-gain", type=float, default=0.0)
    verify.set_defaults(func=_cmd_lineup_verify)
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
