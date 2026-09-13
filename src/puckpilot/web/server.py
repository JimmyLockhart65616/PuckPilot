"""Second-screen web view for draft night.

Deliberately stdlib-only (`http.server`): a draft-night tool should not need a
dependency install, a container, or a build step to come up.

The engine does not pick here. It shows a shortlist with the reasoning on both
sides, the full remaining board, the roster, and what the roster still needs —
and a human decides. Picks arrive on their own from the websocket feed, which
measured 190/190 against a real draft, so there is no manual pick entry to keep
in sync.

The page is the proof as much as the product: it shows how many picks the feed
has seen, whether any pick numbers are missing, and how long since the last one,
so a stalled feed is visible rather than silently frozen.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from puckpilot.draft.advice import market_disagreement, recommend
from puckpilot.draft.board import DraftBoard
from puckpilot.draft.engine import RosterValuePolicy
from puckpilot.draft.explain import summarize

PAGE = """<!doctype html>
<meta charset="utf-8"><title>PuckPilot draft</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#0f1216;--fg:#e8edf2;--dim:#8b98a5;--line:#252b33;--hot:#4ade80;
       --warn:#fbbf24;--bad:#f87171;--card:#161b22}
 @media(prefers-color-scheme:light){:root{--bg:#fff;--fg:#111;--dim:#666;
       --line:#e3e6ea;--card:#f7f8fa}}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:14px}
 h1{font-size:14px;margin:0 0 8px;font-weight:600;color:var(--dim)}
 .bar{display:flex;gap:20px;flex-wrap:wrap;align-items:baseline;
   border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:14px}
 .k{color:var(--dim);font-size:12px} .big{font-size:19px;font-weight:700}
 .live{color:var(--hot)} .stale{color:var(--warn)} .bad{color:var(--bad)}
 .mine{color:var(--hot);font-weight:700}
 .cols{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}
 .col{flex:1;min-width:300px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:6px;
   padding:10px 12px;margin-bottom:10px}
 .nm{font-size:16px;font-weight:700} .meta{color:var(--dim);font-size:12px}
 ul.r{list-style:none;padding:0;margin:6px 0 0} ul.r li{padding:1px 0;font-size:12.5px}
 .pro::before{content:"+ ";color:var(--hot);font-weight:700}
 .con::before{content:"\\2212 ";color:var(--warn);font-weight:700}
 table{border-collapse:collapse;width:100%}
 th{text-align:left;color:var(--dim);font-weight:500;font-size:11px;
   border-bottom:1px solid var(--line);padding:3px 6px 3px 0;position:sticky;top:0;
   background:var(--bg)}
 td{padding:2px 6px 2px 0;border-bottom:1px solid var(--line);font-size:12.5px}
 .num{text-align:right}
 .scroll{max-height:70vh;overflow:auto}
 input{background:transparent;border:1px solid var(--line);color:var(--fg);
   padding:5px 8px;width:100%;font:inherit;border-radius:4px;margin-bottom:6px}
 .need{color:var(--warn)} code{color:var(--dim);font-size:11px}
 .proj{display:flex;flex-wrap:wrap;gap:2px 10px;margin:6px 0 2px;font-size:12px;
   color:var(--dim)} .proj b{color:var(--fg);font-weight:600}
 .gaps{margin-bottom:4px}
 .gap{border-left:2px solid var(--line);padding:3px 0 3px 8px;margin-bottom:6px;font-size:13px}
 .gap .meta{font-size:11.5px;margin-top:1px}
 .gp{color:var(--dim);font-size:11px;margin-right:2px}
 b.us{color:var(--hot)} b.them{color:var(--warn)}
 .thin{color:var(--warn)} .fade{color:var(--dim)}
 button{background:var(--card);border:1px solid var(--line);color:var(--fg);
   font:inherit;padding:4px 10px;border-radius:4px;cursor:pointer}
 button:hover{border-color:var(--warn);color:var(--warn)}
</style>
<h1>PuckPilot</h1>
<div class="bar">
  <div><span class="k">round</span> <span id="round" class="big">-</span></div>
  <div><span class="k">pick</span> <span id="pick" class="big">-</span></div>
  <div><span id="turn"></span></div>
  <div><span class="k">feed</span> <span id="detected" class="big">0</span></div>
  <div><span class="k">last</span> <span id="age">never</span></div>
  <div id="gaps"></div>
  <div><span class="k">left</span> <span id="supply"></span></div>
  <div style="margin-left:auto">
    <button id="undo" title="Take back the last pick the feed recorded">undo last pick</button>
    <span id="undone" class="k"></span></div>
</div>
<div class="cols">
  <div class="col" style="flex:1.15">
    <div class="k">TAKE ONE OF THESE</div>
    <div id="short"></div>
    <div class="k" style="margin-top:10px">YOUR ROSTER</div>
    <div class="card"><div id="roster" class="meta"></div>
      <div id="needs" class="need" style="margin-top:6px"></div></div>
  </div>
  <div class="col" style="flex:0.8;min-width:265px">
    <div class="k">THE ROOM IS SLEEPING ON</div>
    <div id="sleeping" class="gaps"></div>
    <div class="k" style="margin-top:10px">THE ROOM RATES THESE ABOVE US</div>
    <div id="rated" class="gaps"></div>
  </div>
  <div class="col">
    <div class="k">BOARD &mdash; <span id="left">0</span> LEFT</div>
    <input id="filter" placeholder="filter by name or position...">
    <div class="scroll"><table>
      <thead><tr><th>#</th><th>player</th><th>pos</th><th>tm</th>
        <th class="num">vorp</th><th class="num">adp</th><th class="num">lasts</th></tr></thead>
      <tbody id="board"></tbody></table></div>
  </div>
</div>
<p><code id="diag"></code></p>
<script>
let filter = "";
document.getElementById('filter').addEventListener('input', e => {
  filter = e.target.value.toLowerCase(); render(window.__s);
});
function render(s){
  if(!s) return;
  document.getElementById('round').textContent = s.round ?? '-';
  document.getElementById('pick').textContent = (s.made+1)+'/'+s.total;
  document.getElementById('detected').textContent = s.detected;
  document.getElementById('turn').innerHTML = s.my_turn
    ? '<span class="mine">&#9654; YOUR PICK</span>'
    : '<span class="k">seat '+s.on_clock+' &middot; you are up in '+s.picks_away+'</span>';
  const age = s.seconds_since_pick, a = document.getElementById('age');
  a.textContent = age===null ? 'never' : age.toFixed(0)+'s';
  a.className = (age!==null && age < 90) ? 'live' : 'stale';
  document.getElementById('gaps').innerHTML = (s.gaps && s.gaps.length)
    ? '<span class="bad">missing picks '+s.gaps.join(',')+'</span>' : '';

  document.getElementById('supply').innerHTML = (s.supply||[]).map(
    x => '<span class="'+(x[2]?'need':'k')+'" style="margin-right:8px">'+
         x[0]+' <b>'+x[1]+'</b></span>').join('');

  const sh = document.getElementById('short'); sh.innerHTML = '';
  s.shortlist.forEach((c,i)=>{
    const d = document.createElement('div'); d.className='card';
    d.innerHTML = '<div class="nm">'+(i+1)+'. '+c.name+'</div>'+
      '<div class="meta">'+c.position+' &middot; '+c.team+' &middot; VORP '+c.vorp.toFixed(2)+
      ' &middot; ADP '+Math.round(c.adp_rank)+
      ' &middot; lasts '+Math.round(c.p_survive*100)+'%</div>'+
      (c.projected && c.projected.length
        ? '<div class="proj">'+c.projected.map(
            x=>'<span><b>'+x[1]+'</b> '+x[0]+'</span>').join('')+'</div>'
        : '')+
      '<ul class="r">'+c.reasons.map(r=>'<li class="'+r.kind+'">'+r.text+'</li>').join('')+'</ul>';
    sh.appendChild(d);
  });

  document.getElementById('roster').innerHTML = s.roster.length
    ? s.roster.map(p=>'<b>'+p.position+'</b> '+p.name).join(' &middot; ') : '(empty)';
  document.getElementById('needs').textContent = s.needs.length
    ? 'still need: '+s.needs.join(', ') : 'roster minimums met';

  const rows = s.board.filter(p => !filter ||
      p.name.toLowerCase().includes(filter) || p.position.toLowerCase()===filter);
  document.getElementById('left').textContent = s.n_left ?? s.board.length;
  const tb = document.getElementById('board'); tb.innerHTML='';
  rows.slice(0,300).forEach((p,i)=>{
    const tr=document.createElement('tr');
    tr.innerHTML='<td>'+(i+1)+'</td><td>'+p.name+'</td><td>'+p.position+'</td><td>'+p.team+
      '</td><td class="num">'+p.vorp.toFixed(2)+'</td><td class="num">'+Math.round(p.adp_rank)+
      '</td><td class="num">'+Math.round(p.p_survive*100)+'%</td>';
    tb.appendChild(tr);
  });
  const gapRow = (g, mine) =>
    '<div class="gap"><span class="gp">'+g.position+'</span> '+g.name+
    '<div class="meta">we have him <b class="'+(mine?'us':'them')+'">#'+g.our_rank+
    '</b> at '+g.position+', the room has him <b>#'+g.market_rank+'</b>'+
    (g.flag ? ' &middot; <span class="'+(g.flag==='thin history'?'thin':'fade')+'">'+
      g.flag+(g.flag==='thin history' ? ' &mdash; we may be wrong'
                                      : ' &mdash; we may be right')+'</span>' : '')+
    '</div></div>';
  const gaps = s.market_gaps || {sleeping:[], rated:[]};
  document.getElementById('sleeping').innerHTML = gaps.sleeping.length
    ? gaps.sleeping.map(g => gapRow(g, true)).join('')
    : '<div class="meta">nothing startable left that the room is undervaluing</div>';
  document.getElementById('rated').innerHTML = gaps.rated.length
    ? gaps.rated.map(g => gapRow(g, false)).join('')
    : '<div class="meta">no material disagreement</div>';

  document.getElementById('diag').textContent = s.diagnostics;
}
document.getElementById('undo').addEventListener('click', async () => {
  const note = document.getElementById('undone');
  try {
    const r = await fetch('/undo', {method:'POST'});
    note.textContent = (await r.json()).result || '';
  } catch(e) { note.textContent = 'undo failed'; }
  tick();
});
async function tick(){
  try { window.__s = await (await fetch('/state')).json(); render(window.__s); }
  catch(e){ document.getElementById('gaps').innerHTML =
      '<span class="bad">server unreachable</span>'; }
}
tick(); setInterval(tick, 1000);
</script>
"""


def _fmt_cat(value: float, cat) -> str:
    """A counting stat as a whole number, a rate with enough decimals to differ."""
    return f"{value:.3f}".lstrip("0") if getattr(cat, "rate", False) else f"{value:.0f}"


@dataclass
class LiveState:
    """Board plus feed, guarded so HTTP threads and the poller can share it."""

    board: DraftBoard
    policy: RosterValuePolicy = field(default_factory=RosterValuePolicy)
    feed: object | None = None
    top: int = 3
    board_rows: int = 300
    # The league's scored categories, in its own order. Drives both the reason
    # text ("PPP", not "ppp") and the per-category line on each card, so a
    # different league shows its own categories without a code change.
    cats: tuple = ()

    @property
    def labels(self) -> dict[str, str]:
        return {c.key: c.label for c in self.cats}

    lock: threading.Lock = field(default_factory=threading.Lock)
    last_pick_at: float | None = None
    note: str = ""

    def pump(self) -> int:
        """Take whatever the feed has; returns how many picks landed."""
        if self.feed is None:
            return 0
        try:
            events = self.feed.poll(self.board)
        except Exception as e:
            self.note = f"feed error: {e.__class__.__name__}: {e}"
            return 0
        if not events:
            return 0
        from puckpilot.draft.feed import apply

        with self.lock:
            accepted, _rejected = apply(self.board, events)
        if accepted:
            self.last_pick_at = time.time()
        return len(accepted)

    def undo(self) -> str:
        """Recovery hatch: the feed is the only pick source, so a bad frame
        needs a way back without restarting the console mid-draft."""
        with self.lock:
            pick = self.board.undo()
        return f"undid {pick.name}" if pick else "nothing to undo"

    def snapshot(self) -> dict:
        with self.lock:
            cands = recommend(self.board, self.policy, n=max(self.board_rows, 40))
            labels = self.labels
            shortlist = [
                {
                    "name": c.name,
                    "position": c.position,
                    "team": c.team,
                    "vorp": c.vorp,
                    "adp_rank": c.adp_rank,
                    "p_survive": c.p_survive,
                    "reasons": [{"kind": r.kind, "text": r.text} for r in reasons],
                    # What he actually gets you, per category. Already computed
                    # on every Candidate and previously thrown away - it is the
                    # most direct answer to "why is this one better than that
                    # one" when the two look similar on VORP.
                    "projected": [
                        # Rates need their decimals: SV% rounded to an integer
                        # reads "1" and tells the drafter nothing.
                        [c2.label, _fmt_cat(c.projected[c2.key], c2)]
                        for c2 in self.cats
                        if c2.key in c.projected
                    ],
                }
                for c, reasons in summarize(self.board, cands, labels, top=self.top)
            ]
            board_rows = [
                {
                    "name": c.name,
                    "position": c.position,
                    "team": c.team,
                    "vorp": c.vorp,
                    "adp_rank": c.adp_rank,
                    "p_survive": c.p_survive,
                }
                for c in cands[: self.board_rows]
            ]
            seat = self.board.my_seat
            on_clock = self.board.on_the_clock()
            nxt = self.board.next_pick_no(seat)
            roster = [{"name": p.name, "position": p.position} for p in self.board.roster(seat)]
            needs = [f"{p}x{n}" for p, n in sorted(self.board.needs(seat).items())]
            made, total = self.board.made, len(self.board.slots)
            rnd = self.board.current_round()
            # The real number of players still available, not the 300 the table
            # renders - it read "300 LEFT" for most of a draft.
            n_left = int(self.board.avail.sum())
            # Per-position supply, with the positions we still owe starters to
            # marked. The engine does not weight this (see
            # RosterValuePolicy._dynamic_vorp - measured, and left off), so it
            # is surfaced as a fact for the drafter to apply.
            sleeping, rated = market_disagreement(self.board, n=5)
            need_pos = set(self.board.needs(seat))
            supply = [[pos, n, pos in need_pos] for pos, n in sorted(self.board.supply().items())]

        status = self.feed.status() if hasattr(self.feed, "status") else {}
        return {
            "round": rnd,
            "made": made,
            "total": total,
            "on_clock": on_clock,
            "my_turn": on_clock == seat,
            "picks_away": (None if nxt is None else nxt - made),
            "detected": status.get("picks_detected", made),
            "gaps": status.get("gaps", []),
            "n_left": n_left,
            "supply": supply,
            "market_gaps": {
                "sleeping": [asdict(g) for g in sleeping],
                "rated": [asdict(g) for g in rated],
            },
            "seconds_since_pick": (
                None if self.last_pick_at is None else time.time() - self.last_pick_at
            ),
            "shortlist": shortlist,
            "board": board_rows,
            "roster": roster,
            "needs": needs,
            "diagnostics": (
                f"feed={status.get('chosen', 'none')} frames={status.get('frames', 0)} "
                f"unmapped={status.get('unmapped', 0)}" + (f"  |  {self.note}" if self.note else "")
            ),
        }


def make_handler(state: LiveState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console clean
            pass

        def _send(self, code, body, ctype):
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _route(self) -> str:
            """Path without its query string.

            Matched exactly, never by prefix: `startswith("/undo")` also accepts
            `/undo.png`, which is enough to smuggle a state change through an
            `<img>` tag.
            """
            return self.path.split("?", 1)[0].rstrip("/") or "/"

        def _is_same_origin(self) -> bool:
            """Reject cross-site requests to the mutating route.

            A GET or form POST from any other page the user has open is a
            "simple request": the browser sends it and the side effect fires
            even though the same-origin policy hides the reply. The port is a
            fixed default, so the target is guessable. Fetch metadata is the
            primary check; Origin is the fallback for clients that omit it.
            """
            # Host first. Sec-Fetch-Site alone does not survive DNS rebinding:
            # the attacker's own name resolves to 127.0.0.1, so the browser
            # truthfully reports same-origin while the page is theirs. Pinning
            # Host to the loopback names we actually serve closes that, and
            # also protects the read routes.
            # Read the port off the listening socket rather than trusting the
            # value passed in: serve(port=0) lets the OS choose, and comparing
            # against the literal 0 rejected every legitimate request.
            bound = self.server.server_address[1]
            host = (self.headers.get("Host") or "").lower()
            if host and host not in (
                f"127.0.0.1:{bound}",
                f"localhost:{bound}",
                f"[::1]:{bound}",
            ):
                return False
            site = self.headers.get("Sec-Fetch-Site")
            if site is not None:
                return site in ("same-origin", "none")
            origin = self.headers.get("Origin")
            if origin is None:
                return True  # curl and friends: no browser, no CSRF vector
            return origin in (f"http://127.0.0.1:{bound}", f"http://localhost:{bound}")

        def do_GET(self):
            route = self._route()
            if route == "/state":
                # Guarded as well: /state is the entire board, roster and needs,
                # which is exactly what a rebinding attack would want to read.
                if not self._is_same_origin():
                    self._send(
                        403, json.dumps({"error": "cross-origin refused"}), "application/json"
                    )
                    return
                self._send(200, json.dumps(state.snapshot()), "application/json")
            elif route == "/undo":
                # Mutating routes are POST-only, so a bare navigation or an
                # <img> src cannot rewind the board mid-draft.
                self._send(
                    405,
                    json.dumps({"error": "use POST /undo"}),
                    "application/json",
                )
            else:
                self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            if self._route() != "/undo":
                self._send(404, json.dumps({"error": "not found"}), "application/json")
                return
            if not self._is_same_origin():
                self._send(403, json.dumps({"error": "cross-origin refused"}), "application/json")
                return
            self._send(200, json.dumps({"result": state.undo()}), "application/json")

    return Handler


def serve(state: LiveState, port: int = 8765) -> ThreadingHTTPServer:
    """Start the server on a background thread; returns it so callers can stop it."""
    # Loopback only: the board and roster are private, and the server has no auth.
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
