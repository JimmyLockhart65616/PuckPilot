"""The draft-night page itself, kept free of engine imports.

`server.py` pulls in numpy, pandas and scipy through the advice path. The relay
that serves this same page from Azure must not - it holds no board and does no
ranking, it only hands back snapshots the console pushed to it. Keeping the
markup here is what lets both import it.
"""

from __future__ import annotations

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
 tr.blocked td{opacity:.55}
 .blk{color:var(--warn);font-size:10px;border:1px solid var(--line);
   border-radius:3px;padding:0 3px;margin-left:4px;vertical-align:1px}
 .mkt{color:var(--dim);font-size:10px;border:1px solid var(--line);
   border-radius:3px;padding:0 3px}
 button{background:var(--card);border:1px solid var(--line);color:var(--fg);
   font:inherit;padding:4px 10px;border-radius:4px;cursor:pointer}
 button:hover{border-color:var(--warn);color:var(--warn)}
 .warnbar{border:1px solid var(--warn);color:var(--warn);border-radius:6px;padding:6px 10px;
   margin-bottom:10px;font-size:12.5px}
 .warnbar div{padding:1px 0}
 .drift{font-weight:700} .entry{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
 .entry input{width:auto;flex:1;min-width:140px;margin:0}
 .entry input.seatin{flex:0;min-width:64px;width:64px}
 td.act{width:1%;white-space:nowrap} td.act button{padding:0 6px;font-size:11px}
 .src{color:var(--dim);font-size:10.5px}
</style>
<h1>PuckPilot <span id="seatno" class="k"></span></h1>
<div id="warnings" class="warnbar" hidden></div>
<div class="bar">
  <div><span class="k">round</span> <span id="round" class="big">-</span></div>
  <div><span class="k">pick</span> <span id="pick" class="big">-</span></div>
  <div><span id="turn"></span></div>
  <div><span class="k">feed</span> <span id="detected" class="big">0</span></div>
  <div><span class="k">last</span> <span id="age">never</span></div>
  <div><span id="drift" class="drift"></span></div>
  <div id="gaps"></div>
  <div><span class="k">left</span> <span id="supply"></span></div>
  <div style="margin-left:auto">
    <button id="legend-btn" title="What do these numbers mean?">? legend</button>
    <button id="undo" title="Take back the last pick the feed recorded">undo last pick</button>
    <span id="undone" class="k"></span></div>
</div>
<div id="legend" class="card" hidden>
  <div class="k" style="margin-bottom:6px">WHAT THE NUMBERS MEAN &mdash; first time here?</div>
  <ul class="r" style="columns:2 280px;column-gap:24px">
    <li><b>VORP</b> &mdash; value over the last startable player at that position.
      Already accounts for scarcity, so it is the one number safe to compare
      ACROSS positions (a center and a defenseman at the same VORP are equally
      valuable picks). Higher is always better.</li>
    <li><b>ADP</b> &mdash; average draft position: where the room actually
      takes him, not our opinion of him.</li>
    <li><b>lasts %</b> &mdash; odds he is still on the board when it is your
      next turn. High = safe to wait on him; low = take him now or lose him.</li>
    <li><b>TAKE ONE OF THESE</b> &mdash; the engine's top picks right now.
      Each reason is tagged <span class="pro">for taking him</span> or
      <span class="con">against</span>.</li>
    <li><b>THE ROOM IS SLEEPING ON</b> &mdash; startable players we rate well
      above the room's own ADP &mdash; possible value if you wait a beat.</li>
    <li><b>THE ROOM RATES THESE ABOVE US</b> &mdash; players the room takes
      earlier than we would &mdash; a check on our own blind spots, worth a
      second look before you pass.</li>
    <li><b>MARKET ONLY &mdash; NO PROJECTION</b> &mdash; real draft buzz
      (usually a rookie) with no season projection of ours. Priced from the
      room, never from us &mdash; the <span class="mkt">MKT</span> tag marks
      that it is their opinion, not ours.</li>
    <li><b>behind the room</b> &mdash; picks the room has made that this board has
      not recorded. Anything above 0 means every "lasts %" is for a pick that is
      already gone: enter the missing picks by hand (or <b>unknown pick +1</b>).</li>
    <li><b>taken / unknown / kept</b> &mdash; the hand-entry controls. <b>taken</b>
      uses a pick; <b>kept</b> does not (a keeper nobody declared); the <b>x</b>
      on a board row marks exactly that player taken by the seat on the clock.
      <b>undo</b> takes back the last pick however it was entered.</li>
    <li><b>needs</b> &mdash; starting roster slots you still have to fill.</li>
    <li><b>left / supply</b> &mdash; players remaining, overall and by
      position &mdash; a position marked orange is running thin.</li>
    <li><b>blocked <span class="blk">cap</span>/<span class="blk">min</span></b>
      &mdash; your roster rules won't let you draft that position right now
      (already capped, or a minimum elsewhere takes priority). Still shown, not
      hidden &mdash; it's a fact about your roster, not the player.</li>
  </ul>
</div>
<div class="cols">
  <div class="col" style="flex:1.15">
    <div class="k">TAKE ONE OF THESE</div>
    <div id="short"></div>
    <div id="entrybox" hidden>
      <div class="k" style="margin-top:10px">
        ENTER A PICK BY HAND &mdash; when the feed misses one</div>
      <div class="card">
        <div class="entry">
          <input id="who" placeholder="player name (or a board row's x)">
          <input id="by" class="seatin" placeholder="seat"
            title="blank = the seat on the clock">
        </div>
        <div class="entry" style="margin-top:6px">
          <button id="btn-taken" title="Drafted: uses the pick on the clock">taken</button>
          <button id="btn-unknown"
            title="The room took someone we cannot identify: advances the clock one pick"
            >unknown pick +1</button>
          <button id="btn-kept"
            title="A keeper nobody declared: off the board WITHOUT using a pick. Needs a seat."
            >kept (no pick)</button>
        </div>
        <div id="entrynote" class="meta" style="margin-top:6px"></div>
      </div>
    </div>
    <div class="k" style="margin-top:10px">RECENT PICKS</div>
    <div class="card"><div id="recent" class="meta">(none yet)</div></div>
    <div class="k" style="margin-top:10px">YOUR ROSTER</div>
    <div class="card"><div id="roster" class="meta"></div>
      <div id="needs" class="need" style="margin-top:6px"></div></div>
  </div>
  <div class="col" style="flex:0.8;min-width:265px">
    <div class="k">THE ROOM IS SLEEPING ON</div>
    <div id="sleeping" class="gaps"></div>
    <div class="k" style="margin-top:10px">THE ROOM RATES THESE ABOVE US</div>
    <div id="rated" class="gaps"></div>
    <div class="k" style="margin-top:10px">MARKET ONLY &mdash; NO PROJECTION</div>
    <div id="marketonly" class="gaps"></div>
  </div>
  <div class="col">
    <div class="k">BOARD &mdash; <span id="left">0</span> LEFT</div>
    <input id="filter" placeholder="filter by name or position...">
    <div class="scroll"><table>
      <thead><tr><th>#</th><th>player</th><th>pos</th><th>tm</th>
        <th class="num">vorp</th><th class="num">adp</th><th class="num">lasts</th>
        <th></th></tr></thead>
      <tbody id="board"></tbody></table></div>
  </div>
</div>
<p><code id="diag"></code></p>
<script>
// Seat and access key ride in the URL: a shared link is the only thing the
// second manager gets, so it has to carry both. Seat picks the view, not a
// permission - the server gates writes on the key alone.
const P = new URLSearchParams(location.search);
const SEAT = P.get('seat'), KEY = P.get('k');
function q(base){
  const u = new URLSearchParams();
  if(SEAT !== null) u.set('seat', SEAT);
  if(KEY !== null) u.set('k', KEY);
  const s = u.toString();
  return s ? base + '?' + s : base;
}
let filter = "";
const ESC = {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'};
function esc(t){ return String(t ?? '').replace(/[&<>"']/g, c => ESC[c]); }
async function post(route, params){
  // View params (seat, k) ride along exactly as for /state; the pick's own
  // seat is `by`, never `seat` - see server._drafting_seat.
  const u = new URL(q(route), location.href);
  for (const [k, v] of Object.entries(params || {}))
    if (v !== '' && v != null) u.searchParams.set(k, v);
  const note = document.getElementById('entrynote');
  try {
    const r = await fetch(u.pathname + u.search, {method:'POST'});
    const j = await r.json();
    note.textContent = j.result || j.error || '';
    note.className = 'meta ' + ((j.ok === false || j.error) ? 'bad' : 'live');
  } catch(e) { note.textContent = route + ' failed'; note.className = 'meta bad'; }
  tick();
}
document.getElementById('filter').addEventListener('input', e => {
  filter = e.target.value.toLowerCase(); render(window.__s);
});
function render(s){
  if(!s) return;
  document.getElementById('undo').hidden = !s.can_undo;
  // Every board-changing control is owner-only; can_undo is the one flag both
  // the console and the relay already set, so it gates all of them.
  document.getElementById('entrybox').hidden = !s.can_undo;
  const warns = (s.warnings || []);
  const wb = document.getElementById('warnings');
  wb.hidden = !warns.length;
  wb.innerHTML = warns.map(w => '<div>&#9888; ' + esc(w) + '</div>').join('');
  const dr = document.getElementById('drift');
  if (s.drift > 0) {
    dr.className = 'drift bad';
    dr.textContent = s.drift + ' pick' + (s.drift === 1 ? '' : 's') + ' behind the room';
  } else if (s.room_picks) {
    dr.className = 'drift live'; dr.textContent = 'in step with the room';
  } else { dr.className = 'k'; dr.textContent = ''; }
  document.getElementById('recent').innerHTML = (s.recent && s.recent.length)
    ? s.recent.map(p => '#' + p.pick + ' <b>' + esc(p.name) + '</b> ' + esc(p.position) +
        ' &middot; seat ' + p.seat + ' <span class="src">' + esc(p.source) + '</span>').join('<br>')
    : '(none yet)';
  document.getElementById('seatno').textContent = (s.seat === undefined ? '' : 'seat ' + s.seat);
  document.getElementById('round').textContent = s.round ?? '-';
  document.getElementById('pick').textContent = (s.made+1)+'/'+s.total;
  document.getElementById('detected').textContent = s.detected;
  document.getElementById('turn').innerHTML = s.my_turn
    ? '<span class="mine">&#9654; YOUR PICK</span>'
    : '<span class="k">seat '+s.on_clock+' &middot; you are up in '+s.picks_away+'</span>';
  const age = s.seconds_since_pick, a = document.getElementById('age');
  a.textContent = age===null ? 'never' : age.toFixed(0)+'s';
  a.className = (age!==null && age < 90) ? 'live' : 'stale';
  const um = (s.unmapped_names || []);
  document.getElementById('gaps').innerHTML = ((s.gaps && s.gaps.length)
    ? '<span class="bad">missing picks ' + s.gaps.join(',') + '</span> ' : '') +
    (um.length ? '<span class="stale" title="The room took these. They are not on ' +
      'our board, and each used a pick.">not on our board: ' +
      um.map(esc).join(', ') + '</span>' : '');

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
    // A blocked row is shown, not hidden: the rules closing a position is a
    // fact about our roster, not about the player. Dimmed and tagged so it
    // cannot be mistaken for a pick we can make this turn.
    const tag = p.blocked ? ' <span class="blk">'+p.blocked+'</span>' : '';
    if (p.blocked) tr.className = 'blocked';
    tr.innerHTML='<td>'+(i+1)+'</td><td>'+p.name+tag+'</td><td>'+p.position+'</td><td>'+p.team+
      '</td><td class="num">'+p.vorp.toFixed(2)+'</td><td class="num">'+Math.round(p.adp_rank)+
      '</td><td class="num">'+Math.round(p.p_survive*100)+'%</td>'+
      '<td class="act">' + (s.can_undo && p.id != null
        ? '<button data-id="' + p.id + '" title="Mark ' + esc(p.name) +
          ' taken by the seat on the clock">x</button>' : '') + '</td>';
    tb.appendChild(tr);
  });
  const gapRow = (g, mine) =>
    '<div class="gap"><span class="gp">'+g.position+'</span> '+g.name+
    '<div class="meta">we have him <b class="'+(mine?'us':'them')+'">#'+g.our_rank+
    '</b> at '+g.position+', the room has him <b>#'+g.market_rank+'</b>'+
    (g.blocked ? ' <span class="blk">'+g.blocked+'</span>' : '')+
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

  // Priced from the room, not from us - no VORP shown, ever: a market number
  // must never look like an opinion we hold. See draft.market.
  const watch = s.market_watchlist || [];
  document.getElementById('marketonly').innerHTML = watch.length
    ? watch.map(p =>
        '<div class="gap"><span class="gp">'+p.position+'</span> '+p.name+
        '<div class="meta">'+p.team+' &middot; consensus pick <b>~'+Math.round(p.adp_rank)+
        '</b>'+(p.age != null ? ' &middot; age '+p.age.toFixed(1) : '')+
        ' &middot; <span class="mkt">MKT</span></div></div>'
      ).join('')
    : '<div class="meta">nothing left with no projection at all</div>';

  document.getElementById('diag').textContent = s.diagnostics;
}
document.getElementById('board').addEventListener('click', e => {
  const b = e.target.closest('button[data-id]');
  if (b) post('/taken', {player: b.dataset.id});
});
const byVal = () => document.getElementById('by').value.trim();
const whoVal = () => document.getElementById('who').value.trim();
document.getElementById('btn-taken').addEventListener('click', () =>
  post('/taken', {player: whoVal(), by: byVal()}));
document.getElementById('btn-unknown').addEventListener('click', () =>
  post('/unknown', {by: byVal()}));
document.getElementById('btn-kept').addEventListener('click', () =>
  post('/kept', {player: whoVal(), by: byVal()}));
document.getElementById('who').addEventListener('keydown', e => {
  if (e.key === 'Enter') post('/taken', {player: whoVal(), by: byVal()});
});
document.getElementById('legend-btn').addEventListener('click', () => {
  document.getElementById('legend').hidden = !document.getElementById('legend').hidden;
});
document.getElementById('undo').addEventListener('click', async () => {
  const note = document.getElementById('undone');
  try {
    const r = await fetch(q('/undo'), {method:'POST'});
    note.textContent = (await r.json()).result || '';
  } catch(e) { note.textContent = 'undo failed'; }
  tick();
});
async function tick(){
  try { window.__s = await (await fetch(q('/state'))).json(); render(window.__s); }
  catch(e){ document.getElementById('gaps').innerHTML =
      '<span class="bad">server unreachable</span>'; }
}
tick(); setInterval(tick, 1000);
</script>
"""
