"""The in-season view, built for a phone at 6:45pm.

Deliberately not the draft page. That one is a board to study at a desk for
three hours; this is a thing to glance at on the way past, which changes almost
every decision - the swaps come first and everything else is below the fold,
the approve buttons are thumb-sized, and the freshness of the data is stated
rather than implied.

The hardest lesson from the draft relay is the one carried over: a dead console
looked live, because a frozen board and a quiet room render identically. Here a
stale page is worse, because a lineup that was right this morning is wrong by
evening. So the age of the data is at the top, always, and goes loud past a
threshold.

No engine imports, same as the draft relay - this file is copied into a
standard-library-only image.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<title>PuckPilot</title>
<style>
  :root {
    --bg:#0f1115; --card:#171a21; --fg:#e8eaed; --dim:#9aa0a8; --line:#262b34;
    --ok:#3ddc84; --warn:#ffb020; --bad:#ff5c5c; --accent:#5b9dff; --chip:#222732;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg:#f6f7f9; --card:#fff; --fg:#14171c; --dim:#5b616b; --line:#e2e5ea;
      --ok:#0f9d58; --warn:#b26a00; --bad:#c62828; --accent:#1a63d8; --chip:#eef1f6;
    }
  }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body {
    margin:0; background:var(--bg); color:var(--fg);
    font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    padding:0 16px 48px; max-width:720px; margin-inline:auto;
  }
  h1 { font-size:20px; margin:18px 0 4px; }
  h2 { font-size:14px; letter-spacing:.06em; text-transform:uppercase; color:var(--dim);
       margin:26px 0 8px; font-weight:600; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:14px 16px; margin:10px 0; }
  .muted { color:var(--dim); }
  .row { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
  .freshness { position:sticky; top:0; z-index:5; margin:0 -16px 8px; padding:10px 16px;
               background:var(--bg); border-bottom:1px solid var(--line); font-size:13px; }
  .freshness.stale { background:var(--bad); color:#fff; font-weight:600; border-bottom:none; }
  .move { display:flex; gap:10px; align-items:baseline; padding:7px 0;
          border-bottom:1px solid var(--line); }
  .move:last-child { border-bottom:none; }
  .tag { font-size:11px; font-weight:700; letter-spacing:.04em; padding:3px 7px;
         border-radius:6px; background:var(--chip); color:var(--dim); white-space:nowrap; }
  .tag.start { background:rgba(61,220,132,.16); color:var(--ok); }
  .tag.bench { background:rgba(255,92,92,.14); color:var(--bad); }
  .tag.move  { background:rgba(91,157,255,.16); color:var(--accent); }
  .tag.ir, .tag.activate { background:rgba(255,196,0,.16); color:var(--warn); }
  .alert { color:var(--bad); font-weight:600; margin:6px 0; }
  .name { flex:1; }
  .sub { font-size:13px; color:var(--dim); }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  td, th { padding:5px 4px; text-align:right; border-bottom:1px solid var(--line); }
  th:first-child, td:first-child { text-align:left; }
  th { font-size:12px; color:var(--dim); font-weight:600; }
  .close { color:var(--warn); font-weight:600; }
  .gone  { color:var(--dim); text-decoration:line-through; }
  .btns { display:flex; gap:8px; margin-top:10px; }
  button { flex:1; padding:12px; font-size:15px; font-weight:600; border-radius:10px;
           border:1px solid var(--line); background:var(--chip); color:var(--fg); }
  button.yes { background:var(--ok); color:#06210f; border-color:transparent; }
  button.no  { background:transparent; color:var(--dim); }
  button:disabled { opacity:.45; }
  .err { color:var(--bad); }
  .why { margin-top:10px; }
  .why summary { color:var(--accent); font-size:14px; font-weight:600; cursor:pointer; }
  .why-title { font-size:12px; letter-spacing:.04em; text-transform:uppercase;
               color:var(--dim); font-weight:600; margin:12px 0 3px; }
  .why-line { font-size:13px; padding:2px 0; }
</style>
</head>
<body>
<div class="freshness" id="fresh">loading...</div>
<div id="app"></div>

<script>
function q(path) {
  var k = new URLSearchParams(location.search).get('k') || '';
  return path + (path.indexOf('?') < 0 ? '?' : '&') + 'k=' + encodeURIComponent(k);
}
function el(tag, cls, text) {
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}
function card(parent) { var c = el('div', 'card'); parent.appendChild(c); return c; }
function h2(parent, t) { parent.appendChild(el('h2', null, t)); }

function ageText(sec) {
  if (sec === null || sec === undefined) return 'never updated';
  if (sec < 90) return 'updated just now';
  if (sec < 5400) return 'updated ' + Math.round(sec / 60) + ' min ago';
  return 'updated ' + Math.round(sec / 3600) + ' h ago';
}

function renderMoves(root, s) {
  h2(root, 'Tonight' + (s.date ? ' \\u00b7 ' + s.date : ''));
  var c = card(root);
  (s.alerts || []).forEach(function (a) { c.appendChild(el('div', 'alert', a)); });
  var moves = s.moves || [];
  if (!moves.length) {
    c.appendChild(el('div', null, s.playing ? 'Lineup is already right.' : 'Nothing to do.'));
  } else {
    moves.forEach(function (m) {
      var row = el('div', 'move');
      var kind = m.kind || 'move';
      row.appendChild(el('span', 'tag ' + kind, kind.toUpperCase()));
      var name = el('span', 'name', m.name);
      if (m.detail) name.appendChild(el('div', 'sub', m.detail));
      row.appendChild(name);
      c.appendChild(row);
    });
  }
  var bits = [];
  if (s.lock_local) bits.push('locks ' + s.lock_local);
  if (s.playing !== undefined) bits.push(s.playing + ' of ' + s.rostered + ' play');
  if (bits.length) c.appendChild(el('div', 'sub', bits.join(' \\u00b7 ')));
  if (s.out && s.out.length) c.appendChild(el('div', 'sub', 'Out: ' + s.out.join(', ')));
}

function decide(kind, id, approve, btn) {
  var box = btn.parentNode;
  Array.prototype.forEach.call(box.querySelectorAll('button'), function (b) { b.disabled = true; });
  fetch(q('/decide'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: kind, id: id, approve: approve })
  }).then(function (r) { return r.json(); }).then(function (out) {
    box.parentNode.replaceChild(
      el('div', 'sub', out.error ? 'Failed: ' + out.error
                                 : (approve ? 'Approved.' : 'Rejected.') +
                                   ' Applied on the next run.'),
      box);
  }).catch(function () {
    box.parentNode.replaceChild(el('div', 'sub err', 'Could not reach the server.'), box);
  });
}

function decisionButtons(c, kind, id) {
  var box = el('div', 'btns');
  var yes = el('button', 'yes', 'Approve');
  var no = el('button', 'no', 'Reject');
  yes.onclick = function () { decide(kind, id, true, yes); };
  no.onclick = function () { decide(kind, id, false, no); };
  box.appendChild(yes); box.appendChild(no);
  c.appendChild(box);
}

function renderProtocol(root, p) {
  if (!p) return;
  h2(root, 'This week\\u2019s plan');
  var c = card(root);
  c.appendChild(el('div', null, 'Week ' + p.week + (p.opponent ? ' vs ' + p.opponent : '')));
  (p.give_up || []).forEach(function (t) {
    c.appendChild(el('div', 'sub', 'Give up ' + t));
  });
  (p.go_after || []).forEach(function (t) {
    c.appendChild(el('div', 'sub', 'Go after ' + t));
  });
  if (p.status === 'proposed') decisionButtons(c, 'protocol', p.id);
  else c.appendChild(el('div', 'sub', p.status));
}

function renderProposals(root, list) {
  if (!list || !list.length) return;
  h2(root, 'Needs your decision');
  list.forEach(function (p) {
    var c = card(root);
    c.appendChild(el('div', null, 'Add ' + p.add + (p.drop ? '  \\u2013  drop ' + p.drop : '')));
    if (p.why) c.appendChild(el('div', 'sub', p.why));
    if (p.timing) c.appendChild(el('div', 'sub', p.timing));
    renderReasons(c, p);
    decisionButtons(c, 'proposal', p.id);
  });
}

// The page redraws every 20 s; an opened "why" has to stay open through it.
var opened = {};
function renderReasons(c, p) {
  var sections = p.detail || [];
  if (!sections.length) return;
  var d = el('details', 'why');
  d.open = !!opened[p.id];
  d.addEventListener('toggle', function () { opened[p.id] = d.open; });
  d.appendChild(el('summary', null, 'Why \\u2013 day by day, odds, ranges, season'));
  sections.forEach(function (s) {
    d.appendChild(el('div', 'why-title', s.title));
    (s.lines || []).forEach(function (line) { d.appendChild(el('div', 'why-line', line)); });
  });
  c.appendChild(d);
}

function pair(a, b) {
  if (a === undefined || a === null || b === undefined || b === null) return '–';
  return a + '–' + b;
}

function renderWeek(root, w) {
  if (!w || !w.cats || !w.cats.length) return;
  h2(root, 'Week ' + (w.week || '') + (w.opponent ? ' vs ' + w.opponent : ''));
  var c = card(root);
  if (w.expected !== undefined && w.expected !== null) {
    c.appendChild(el('div', null, 'Expect ' + w.expected + ' of ' + w.of + ' categories'));
  }
  var gl = w.games_left;
  if (gl && gl.ours !== undefined) {
    var bits = ['Starts left: you ' + gl.ours + ', them ' + gl.theirs];
    if (w.days_left) bits.push(w.days_left + ' day' + (w.days_left === 1 ? '' : 's') + ' to play');
    c.appendChild(el('div', 'sub', bits.join(' · ')));
  }
  // Once the week has begun, what is banked sits beside where it should end.
  var live = w.cats.some(function (r) { return r.now_ours !== undefined && r.now_ours !== null; });
  var odds = w.cats.some(function (r) { return r.chance !== undefined && r.chance !== null; });
  var t = el('table');
  var head = el('tr');
  var cols = live ? ['cat', 'now', 'final'] : ['cat', 'you', 'them'];
  if (odds) cols.push('win');
  cols.push('');
  cols.forEach(function (x) { head.appendChild(el('th', null, x)); });
  t.appendChild(head);
  w.cats.forEach(function (row) {
    var tr = el('tr');
    tr.appendChild(el('td', row.state === 'gone' ? 'gone' : null, row.label));
    if (live) {
      tr.appendChild(el('td', null, pair(row.now_ours, row.now_theirs)));
      tr.appendChild(el('td', null, pair(row.ours, row.theirs)));
    } else {
      tr.appendChild(el('td', null, row.ours));
      tr.appendChild(el('td', null, row.theirs));
    }
    if (odds) {
      var ch = row.chance === undefined || row.chance === null ? '–' : row.chance + '%';
      tr.appendChild(el('td', null, ch));
    }
    var hot = row.state === 'in play' || row.state === 'close';
    tr.appendChild(el('td', hot ? 'close' : 'muted', row.state));
    t.appendChild(tr);
  });
  c.appendChild(t);
  if (w.note) c.appendChild(el('div', 'sub', w.note));
}

function renderRoster(root, rows) {
  if (!rows || !rows.length) return;
  h2(root, 'Roster');
  var c = card(root);
  rows.forEach(function (r) {
    var row = el('div', 'move');
    row.appendChild(el('span', 'tag', r.slot));
    var n = el('span', 'name', r.name);
    n.appendChild(el('div', 'sub', [r.team, r.opp, r.status].filter(Boolean).join(' \\u00b7 ')));
    row.appendChild(n);
    c.appendChild(row);
  });
}

function render(s) {
  var fresh = document.getElementById('fresh');
  var next = s.next_local ? ' \\u00b7 next ' + s.next_local : '';
  fresh.textContent = ageText(s.age_seconds) + next + (s.manager ? ' \\u00b7 ' + s.manager : '');
  fresh.className = 'freshness' + (s.stale ? ' stale' : '');
  if (s.stale) {
    fresh.textContent = 'NOT LIVE \\u2013 ' + ageText(s.age_seconds) +
      (s.next_local ? ' \\u00b7 a run was due ' + s.next_local : '');
  }

  var app = document.getElementById('app');
  app.textContent = '';
  if (s.empty) {
    app.appendChild(el('h1', null, 'PuckPilot'));
    card(app).appendChild(el('div', 'muted', 'Nothing pushed yet today.'));
    return;
  }
  app.appendChild(el('h1', null, s.team || 'PuckPilot'));
  renderMoves(app, s);
  renderProposals(app, s.proposals);
  renderProtocol(app, s.protocol);
  renderWeek(app, s.week);
  renderRoster(app, s.roster);
}

var failures = 0;
function tick() {
  fetch(q('/state')).then(function (r) {
    if (!r.ok) throw new Error(r.status === 403 ? 'wrong or missing key' : 'server ' + r.status);
    return r.json();
  }).then(function (s) {
    failures = 0;
    window.__s = s;
    try { render(s); } catch (e) {
      document.getElementById('app').textContent = 'Could not draw this page: ' + e.message;
    }
  }).catch(function (e) {
    failures += 1;
    if (failures > 1) {
      var f = document.getElementById('fresh');
      f.className = 'freshness stale';
      f.textContent = 'NOT LIVE \\u2013 ' + e.message;
    }
  });
}
tick();
setInterval(tick, 20000);
document.addEventListener('visibilitychange', function () {
  if (!document.hidden) tick();
});
</script>
</body>
</html>
"""
