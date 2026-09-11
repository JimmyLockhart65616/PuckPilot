"""Watch a draft happen, to settle whether `draftresults` is live.

The whole session-API feed rests on one unverified assumption: that
`league/{key}/draftresults` fills in *during* a draft rather than only once it
finishes. Completed drafts prove the payload parses; they say nothing about
timing, because they are all `postdraft` by the time we see them.

So this polls the endpoint through a draft and records, per poll, how many picks
are visible and how long after the previous one they appeared. Two outcomes:

- counts climb while `draft_status` is `drafting` -> the feed works live, and the
  observed lag is the real latency budget for the draft-night console;
- counts stay at 0 until the draft ends and then jump to full -> the feed is
  useless on the night, and the pick source has to come from the draft room
  itself (which is what `ppilot draft capture` is for).

Either way the answer is empirical and written to disk, not assumed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Polling budget. Each pass costs two requests against an undocumented endpoint,
# so this is capped rather than left as a function of how long the draft runs.
# A pick takes tens of seconds; 20s spacing detects liveness with room to spare.
DEFAULT_INTERVAL_S = 20.0
MAX_REQUESTS = 400

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


@dataclass
class WatchSample:
    t: float
    wall: str
    draft_status: str
    n_picks: int
    error: str | None = None


@dataclass
class WatchReport:
    league_key: str
    samples: list[WatchSample] = field(default_factory=list)
    first_pick_seen_at: float | None = None
    statuses: set[str] = field(default_factory=set)
    note: str = ""

    @property
    def verdict(self) -> str:
        saw_drafting = "drafting" in self.statuses
        grew_while_drafting = any(
            s.n_picks > 0 and s.draft_status == "drafting" for s in self.samples
        )
        if grew_while_drafting:
            return "LIVE: picks appeared while draft_status was 'drafting'."
        if saw_drafting:
            return (
                "NOT LIVE: draft_status reached 'drafting' but no picks were ever "
                "visible during it. The draft-night feed cannot use draftresults."
            )
        return "INCONCLUSIVE: never observed draft_status 'drafting'."

    @property
    def text(self) -> str:
        counts = [s.n_picks for s in self.samples]
        lines = [
            f"Draft watch: {self.league_key}",
            f"  samples {len(self.samples)}   statuses seen: {sorted(self.statuses)}",
            f"  picks visible: min {min(counts, default=0)} -> max {max(counts, default=0)}",
        ]
        # the moment a pick first showed up is the whole point
        growth = [
            (b.t, b.n_picks - a.n_picks)
            for a, b in zip(self.samples, self.samples[1:], strict=False)
            if b.n_picks != a.n_picks
        ]
        if growth:
            lines.append(f"  first change at t={growth[0][0]:.1f}s; {len(growth)} increments")
            gaps = [b - a for (a, _), (b, _) in zip(growth, growth[1:], strict=False)]
            if gaps:
                lines.append(
                    f"  median gap between increments: {sorted(gaps)[len(gaps) // 2]:.1f}s"
                )
        if self.note:
            lines.append(f"  {self.note}")
        lines.append("")
        lines.append(f"VERDICT: {self.verdict}")
        return "\n".join(lines)


def watch(
    session,
    league_key: str,
    interval: float = DEFAULT_INTERVAL_S,
    duration: float = 3600.0,
    out_path: Path | None = None,
    progress: Progress = _noop,
    max_requests: int = MAX_REQUESTS,
) -> WatchReport:
    """Poll until the draft completes, the request budget runs out, `duration`
    elapses, or Ctrl+C.

    The budget is the point. Each pass costs two requests against an
    undocumented endpoint, and an earlier 3-second default over a 90-minute
    draft came to roughly 3,600 of them - which is a robot by any reading, and
    is the pattern `session.py`'s own rules forbid. A draft pick takes tens of
    seconds, so polling far slower loses nothing and the ceiling makes the worst
    case explicit rather than a function of how long the draft ran.
    """
    report = WatchReport(league_key=league_key)
    started = time.time()
    last_n = -1
    spent = 0

    progress(f"Watching {league_key}. Ctrl+C to stop.")
    progress(f"  budget: {max_requests} requests, {interval:.0f}s apart")
    try:
        while time.time() - started < duration:
            if spent + 2 > max_requests:
                progress(f"  request budget spent ({spent}); stopping.")
                report.note = f"stopped after {spent} requests (budget {max_requests})"
                break
            t = time.time() - started
            status, n, err = "?", 0, None
            try:
                spent += 2
                status = str(session.league_meta(league_key).get("draft_status", "?"))
                n = len(session.draft_results(league_key))
            except Exception as e:  # a blip must not end the experiment
                err = f"{e.__class__.__name__}: {e}"
            sample = WatchSample(
                t=round(t, 2),
                wall=datetime.now(UTC).isoformat(),
                draft_status=status,
                n_picks=n,
                error=err,
            )
            report.samples.append(sample)
            report.statuses.add(status)
            if n > 0 and report.first_pick_seen_at is None:
                report.first_pick_seen_at = t
            if n != last_n or err:
                progress(f"  [{t:>6.1f}s] status={status:<10} picks={n}{'  ' + err if err else ''}")
                last_n = n
            if status == "postdraft" and n > 0:
                progress("  draft complete.")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        progress("  stopped.")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "league_key": league_key,
                    "verdict": report.verdict,
                    "samples": [vars(s) for s in report.samples],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return report
