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
 button{background:var(--card);border:1px solid var(--line);color:var(--fg);
   font:inherit;padding:4px 10px;border-radius:4px;cursor:pointer}
 button:hover{border-color:var(--warn);color:var(--warn)}
</style>
<h1>PuckPilot <span id="seatno" class="k"></span></h1>
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
document.getElementById('filter').addEventListener('input', e => {
  filter = e.target.value.toLowerCase(); render(window.__s);
});
function render(s){
  if(!s) return;
  document.getElementById('undo').hidden = !s.can_undo;
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
    // A blocked row is shown, not hidden: the rules closing a position is a
    // fact about our roster, not about the player. Dimmed and tagged so it
    // cannot be mistaken for a pick we can make this turn.
    const tag = p.blocked ? ' <span class="blk">'+p.blocked+'</span>' : '';
    if (p.blocked) tr.className = 'blocked';
    tr.innerHTML='<td>'+(i+1)+'</td><td>'+p.name+tag+'</td><td>'+p.position+'</td><td>'+p.team+
      '</td><td class="num">'+p.vorp.toFixed(2)+'</td><td class="num">'+Math.round(p.adp_rank)+
      '</td><td class="num">'+Math.round(p.p_survive*100)+'%</td>';
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

  document.getElementById('diag').textContent = s.diagnostics;
}
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
