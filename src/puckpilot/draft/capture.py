"""DIAGNOSTIC ONLY: read-only observation harness for the Yahoo draft room.

Not part of the draft-night path and not a data source for any engine. This
exists to answer one question empirically - where do picks come from? - and it
answered it: the draft room broadcasts them on a websocket
(`draft/wsfeed.py`, measured 190/190). It is kept as the instrument to re-run if
Yahoo changes that protocol, not as something anything depends on.

Conduct rules it holds to:

- **It never drives the browser.** It attaches to a session the user is driving
  by hand and listens. It does not click, type, submit, or navigate on its own.
- **It observes only what that user's own browser already receives.** No extra
  requests are issued, so it adds nothing to Yahoo's load.
- **Nothing is republished.** Output is local and git-ignored. Sensitive headers
  are redacted by name, and `scrub` additionally blanks credential-shaped values
  anywhere else they appear - query strings, request and response bodies,
  websocket frames in both directions, and the DOM snapshots. That is a net
  rather than a guarantee, which is why the output stays git-ignored regardless
  (this repo is public).

`yahoo/client.py` (OAuth) remains the supported integration. Retire this the day
scraping stops being the only way to see a draft happen.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Redacted rather than dropped: knowing that a request *carried* a cookie is
# useful, the value is not.
SENSITIVE_HEADERS = {
    "cookie",
    "set-cookie",
    "authorization",
    "proxy-authorization",
    "x-csrf-token",
    "y-rid",
}
# Page furniture that cannot carry pick data; dropping it keeps the log readable.
SKIP_RESOURCE_TYPES = {"image", "font", "stylesheet", "media", "other"}
# Only fetch bodies that could plausibly be a data feed.
BODY_CONTENT_HINTS = ("json", "javascript", "text/plain", "xml", "protobuf")
MAX_BODY_BYTES = 256_000
# Pages worth snapshotting; a browser being driven by hand has other tabs open.
# `draftclient` is listed first and explicitly: it is the page this harness
# exists for, and it currently matches "sports.yahoo.com" only by the accident
# that the host is fantasy*sports*.yahoo.com. Relying on that is how the wrong
# tab gets watched.
PAGE_URL_HINTS = (
    "draftclient",
    "fantasysports.yahoo.com",
    "fantasy.yahoo.com",
    "sports.yahoo.com",
    "yahoo.com/fantasy",
)
# A Yahoo fantasy page is mostly ad exchanges: an 18-second probe of the mock
# lobby logged 3,787 requests, and every one of the top twelve hosts was an ad
# network. Keeping only Yahoo's own hosts is the difference between a log that
# can be read and one that cannot. Override with capture(all_hosts=True).
HOST_ALLOW = ("yahoo.com", "yahooapis.com", "yimg.com", "yahoosandbox.com")
# ...but never drop something that looks like the draft itself, whatever host it
# is served from. The draft client is the one thing this harness exists to see,
# and assuming it lives on a Yahoo host is exactly the kind of guess that has
# already cost us two captures.
URL_ALWAYS_KEEP = ("draft", "mock", "/pick", "roster")

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _redact(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: ("<redacted>" if k.lower() in SENSITIVE_HEADERS else v)
        for k, v in (headers or {}).items()
    }


# Headers were once the ONLY thing redacted, which left the more likely carriers
# untouched: a crumb in a query string, a token in a JSON response body, and the
# frames the browser SENDS to a websocket - which is exactly where a client-side
# auth handshake appears. `scrub` runs over every string written to disk.
_KEYS = r"access_token|refresh_token|id_token|crumb|password|secret|sessionid|auth|sig|token"
# `\\?` before every quote: these run over JSON-serialized lines as well as raw
# values, and a nested body arrives escaped - `\"access_token\": \"ya29...\"`.
# Without tolerating the backslash the escaped form matched nothing, which was
# verified leaking a token through the serialized path.
_Q = r"\\?[\"']"
SECRET_PATTERNS = (
    re.compile(rf"(?i)\b({_KEYS})=([^&\s\"'\\]{{6,}})"),
    re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._\-]{12,})"),
    re.compile(rf"(?i){_Q}?({_KEYS}){_Q}?\s*:\s*{_Q}([^\"'\\]{{6,}}){_Q}"),
    # bare JWTs, wherever they appear
    re.compile(r"\b(eyJ)[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]*"),
)


def scrub(text: str) -> str:
    """Blank out credential-shaped substrings anywhere in a captured string.

    Deliberately conservative about what it keeps: the key name survives so the
    log still shows that a token was present, only the value goes. It is a net,
    not a guarantee - which is why captures stay git-ignored regardless.
    """
    if not text:
        return text
    out = text
    for pattern in SECRET_PATTERNS[:3]:
        out = pattern.sub(lambda m: f"{m.group(1)}=<redacted>", out)
    out = SECRET_PATTERNS[3].sub("<redacted-jwt>", out)
    return out


def _on_allowed_host(url: str) -> bool:
    low = url.lower()
    if any(hint in low for hint in URL_ALWAYS_KEEP):
        return True
    host = url.split("/")[2].lower() if "://" in url else ""
    return any(host == h or host.endswith("." + h) for h in HOST_ALLOW)


def _wants_body(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return any(h in ct for h in BODY_CONTENT_HINTS)


# The draft room header carries an authoritative pick counter and the most
# recent pick: "<manager>'s Pick - You're up in 1 Picks - Round 5, Pick 67" and
# "Last: J. SANDERSON (D-OTT)". Both are read by regex over the page's text
# rather than by CSS selector, because Yahoo's atomic class names are not
# stable identifiers and guessing at them is what broke the last two attempts.
# The counter matters most: it is ground truth for how many picks have happened,
# which makes a missed pick detectable instead of silent.
ROUND_PICK_RE = re.compile(r"Round\s+(\d+)\s*[,•·|-]*\s*Pick\s+(\d+)", re.I)
LAST_PICK_RE = re.compile(r"Last:\s*([^\n\r]{2,60})", re.I)
UP_IN_RE = re.compile(r"up\s+in\s+(\d+)\s+Pick", re.I)

VALUES_JS = "() => (document.body ? document.body.innerText : '')"


def dom_values(text: str) -> dict:
    """Pick counter and last pick, parsed out of a draft room's visible text."""
    out: dict = {}
    if not text:
        return out
    rp = ROUND_PICK_RE.search(text)
    if rp:
        out["round"] = int(rp.group(1))
        out["pick"] = int(rp.group(2))
    last = LAST_PICK_RE.search(text)
    if last:
        out["last_pick"] = last.group(1).strip()
    up = UP_IN_RE.search(text)
    if up:
        out["picks_until_mine"] = int(up.group(1))
    return out


@dataclass
class CaptureSession:
    """One recording run: a directory of JSONL streams plus DOM snapshots."""

    out_dir: Path
    started_at: float = field(default_factory=time.time)
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        (self.out_dir / "dom").mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Any] = {}

    def _stream(self, name: str):
        if name not in self._files:
            self._files[name] = (self.out_dir / f"{name}.jsonl").open(
                "a", encoding="utf-8", buffering=1
            )
        return self._files[name]

    def write(self, stream: str, record: dict) -> None:
        """Append one record. Everything is scrubbed HERE, at the chokepoint.

        Scrubbing at each call site was tried and failed the obvious way: six
        paths were covered and twelve were not, including the exception text in
        `dom_snapshot_failed` and the manifest's record of the local profile
        path. Doing it on the serialized line means a write path added later
        cannot bypass it by omission.
        """
        record["t"] = round(time.time() - self.started_at, 3)
        record["wall"] = datetime.now(UTC).isoformat()
        line = scrub(json.dumps(record, ensure_ascii=False))
        self._stream(stream).write(line + "\n")
        self.counts[stream] = self.counts.get(stream, 0) + 1

    def snapshot_dom(self, page, seq: int) -> None:
        try:
            html = page.content()
        except Exception as e:  # page navigating or closed mid-snapshot
            self.write("events", {"kind": "dom_snapshot_failed", "error": str(e)})
            return
        name = f"{seq:04d}_{int(time.time() - self.started_at):06d}.html"
        # A rendered Yahoo page embeds its own bootstrap state, which is where a
        # crumb or token sits in the markup rather than in a header.
        (self.out_dir / "dom" / name).write_text(scrub(html), encoding="utf-8")
        self.write("events", {"kind": "dom_snapshot", "file": name, "url": scrub(page.url)})

    def finalize(self, manifest: dict) -> None:
        manifest["counts"] = self.counts
        manifest["duration_s"] = round(time.time() - self.started_at, 1)
        # The manifest records the attach mode and the local profile path.
        (self.out_dir / "manifest.json").write_text(
            scrub(json.dumps(manifest, indent=2, ensure_ascii=False)), encoding="utf-8"
        )
        for f in self._files.values():
            f.close()
        self._files.clear()


def _attach_page(
    session: CaptureSession, page, progress: Progress, all_hosts: bool = False
) -> None:
    """Wire read-only listeners onto one page."""
    page_url = page.url

    def keep(url: str) -> bool:
        return all_hosts or _on_allowed_host(url)

    def on_request(request) -> None:
        if request.resource_type in SKIP_RESOURCE_TYPES or not keep(request.url):
            return
        session.write(
            "network",
            {
                "kind": "request",
                "method": request.method,
                "url": scrub(request.url),
                "resource_type": request.resource_type,
                "headers": _redact(request.headers),
                "post_data": scrub((request.post_data or "")[:MAX_BODY_BYTES]),
            },
        )

    def on_response(response) -> None:
        request = response.request
        if request.resource_type in SKIP_RESOURCE_TYPES or not keep(response.url):
            return
        headers = {}
        with contextlib.suppress(Exception):
            headers = response.headers
        content_type = headers.get("content-type", "")
        body = None
        if _wants_body(content_type):
            try:
                raw = response.body()
                body = scrub(raw[:MAX_BODY_BYTES].decode("utf-8", errors="replace"))
            except Exception as e:
                body = f"<unavailable: {e}>"
        session.write(
            "network",
            {
                "kind": "response",
                "status": response.status,
                "url": scrub(response.url),
                "resource_type": request.resource_type,
                "content_type": content_type,
                "headers": _redact(headers),
                "body": body,
            },
        )

    def on_websocket(ws) -> None:
        # The highest-value target: a draft room that gets picks pushed will show
        # them here, in order, with arrival times.
        session.write("events", {"kind": "websocket_open", "url": ws.url})
        ws.on(
            "framereceived",
            lambda payload: session.write(
                "websocket",
                {"dir": "recv", "url": ws.url, "payload": scrub(str(payload)[:MAX_BODY_BYTES])},
            ),
        )
        ws.on(
            "framesent",
            lambda payload: session.write(
                "websocket",
                {"dir": "sent", "url": ws.url, "payload": scrub(str(payload)[:MAX_BODY_BYTES])},
            ),
        )
        ws.on(
            "close",
            lambda _: session.write("events", {"kind": "websocket_close", "url": ws.url}),
        )

    page.on("request", on_request)
    page.on("response", on_response)
    page.on("websocket", on_websocket)
    page.on(
        "framenavigated",
        lambda frame: (
            session.write("events", {"kind": "navigated", "url": frame.url})
            if frame == page.main_frame
            else None
        ),
    )
    session.write("events", {"kind": "page_attached", "url": page_url})
    progress(f"  attached to page: {page_url or '(blank)'}")


class ProfileInUse(RuntimeError):
    """The capture profile is already open in another Chrome."""


def profile_is_busy(user_data_dir: Path) -> bool:
    """True when another Chrome already holds this profile.

    Chrome refuses a second instance on one profile, and Playwright surfaces
    that as an opaque launch failure. It is a genuinely easy mistake to make -
    starting a second capture while the first is still counting down does it -
    so it gets detected and named rather than left as a traceback.
    """
    if not user_data_dir.exists():
        return False
    # Chrome's ProcessSingleton: a lock on Windows, a symlink elsewhere.
    for marker in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
        if (user_data_dir / marker).exists():
            return True
    lockfile = user_data_dir / "lockfile"
    if lockfile.exists():
        try:  # if we can rename it, nothing holds it open
            lockfile.rename(lockfile)
        except OSError:
            return True
    return False


def _interesting(page) -> bool:
    return any(h in (page.url or "") for h in PAGE_URL_HINTS)


def run_capture(
    out_dir: Path,
    cdp_url: str | None = None,
    user_data_dir: Path | None = None,
    url: str | None = None,
    dom_interval: float = 15.0,
    values_interval: float = 1.5,
    duration: float | None = None,
    all_hosts: bool = False,
    progress: Progress = _noop,
) -> Path:
    """Record a draft room until interrupted. Returns the session directory.

    Attach order matters. CDP attaches to the user's own already-running Chrome,
    where they are logged in and which presents no automation surface for Yahoo
    to object to; a persistent profile is next; bundled Chromium is last because
    Yahoo login is the likeliest thing to break there.
    """
    import asyncio

    from playwright.sync_api import sync_playwright

    # Closing the browser leaves in-flight listener callbacks whose futures
    # asyncio then reports as "Future exception was never retrieved" - dozens of
    # tracebacks that bury the actual summary at exactly the moment the user is
    # looking for it. The recording is already finalized by then.
    def _quiet_teardown(_loop, context):
        if isinstance(context.get("exception"), Exception):
            return
        _loop.default_exception_handler(context)

    with contextlib.suppress(RuntimeError):
        asyncio.get_event_loop().set_exception_handler(_quiet_teardown)

    session = CaptureSession(out_dir=out_dir)
    manifest: dict[str, Any] = {
        "started": datetime.now(UTC).isoformat(),
        "dom_interval_s": dom_interval,
        "values_interval_s": values_interval,
        "all_hosts": all_hosts,
        "host_allow": None if all_hosts else list(HOST_ALLOW),
    }
    attached: set[int] = set()

    if user_data_dir and profile_is_busy(user_data_dir):
        session.finalize({**manifest, "stop_reason": "profile already in use"})
        raise ProfileInUse(
            "\n".join(
                [
                    f"{user_data_dir} is already open in another Chrome.",
                    "Close that window, and let any running `draft capture` finish first.",
                    "Only one capture can hold the profile at a time.",
                ]
            )
        )

    try:
        with sync_playwright() as p:
            if cdp_url:
                progress(f"Attaching over CDP to {cdp_url} ...")
                browser = p.chromium.connect_over_cdp(cdp_url)
                contexts = browser.contexts
                manifest["attach"] = {"mode": "cdp", "url": cdp_url}
            elif user_data_dir:
                progress(f"Launching Chrome with persistent profile {user_data_dir} ...")
                ctx = p.chromium.launch_persistent_context(
                    str(user_data_dir), headless=False, channel="chrome"
                )
                contexts = [ctx]
                manifest["attach"] = {"mode": "persistent", "user_data_dir": str(user_data_dir)}
            else:
                progress("Launching bundled Chromium (Yahoo login may resist this) ...")
                browser = p.chromium.launch(headless=False)
                contexts = [browser.new_context()]
                manifest["attach"] = {"mode": "bundled"}

            def attach(page) -> None:
                if id(page) in attached:
                    return
                attached.add(id(page))
                _attach_page(session, page, progress, all_hosts)

            for ctx in contexts:
                ctx.on("page", attach)
                for page in ctx.pages:
                    attach(page)

            if not any(ctx.pages for ctx in contexts):
                attach(contexts[0].new_page())

            if url:
                # Convenience only: opens the lobby so the user does not have to
                # find it. Everything after this is driven by hand.
                target = next((pg for ctx in contexts for pg in ctx.pages), None)
                if target is not None:
                    progress(f"Opening {url} ...")
                    try:
                        target.goto(url, wait_until="domcontentloaded")
                    except Exception as e:
                        # Never abort a recording over navigation: the user can just
                        # browse there by hand, and the listeners are already live.
                        progress(f"  could not open it ({e.__class__.__name__}); navigate manually")
                        session.write(
                            "events", {"kind": "goto_failed", "url": url, "error": str(e)}
                        )

            progress("")
            progress("Recording. Open the Yahoo mock draft and play it out normally.")
            progress("Nothing here will touch the draft — it only watches.")
            progress("Press Ctrl+C when the draft is done.")
            progress("")

            seq = 0
            last_snapshot = 0.0
            last_values: dict[int, dict] = {}
            deadline = None if duration is None else time.time() + duration
            stop_reason = "duration elapsed"
            try:
                while deadline is None or time.time() < deadline:
                    # Blocking here is also what lets Playwright dispatch the
                    # request/response/websocket callbacks registered above.
                    pages = [pg for ctx in contexts for pg in ctx.pages]
                    target = next(
                        (pg for pg in pages if _interesting(pg)), pages[0] if pages else None
                    )
                    if target is None:
                        stop_reason = "no pages left open"
                        break

                    # Fast lane: the header's pick counter and last pick, every
                    # values_interval. This is a candidate pick source in its own
                    # right and the ground truth for how many picks have actually
                    # happened, so it is sampled far more often than the heavy
                    # full-page HTML snapshot.
                    target.wait_for_timeout(values_interval * 1000)
                    for page in pages[:3]:
                        try:
                            values = dom_values(page.evaluate(VALUES_JS))
                        except Exception:
                            continue  # tab closed or navigating; not fatal
                        if not values:
                            continue
                        if values != last_values.get(id(page)):
                            last_values[id(page)] = values
                            session.write("dom_values", {"url": page.url.split("?")[0], **values})

                    if time.time() - last_snapshot >= dom_interval:
                        last_snapshot = time.time()
                        seq += 1
                        session.snapshot_dom(target, seq)
                    progress(
                        f"  [{int(time.time() - session.started_at):>5}s] "
                        f"net={session.counts.get('network', 0)} "
                        f"ws={session.counts.get('websocket', 0)} "
                        f"vals={session.counts.get('dom_values', 0)} "
                        f"dom={seq}"
                    )
            except KeyboardInterrupt:
                stop_reason = "interrupted (Ctrl+C)"
                progress("")
                progress("Stopping (Ctrl+C).")
            except Exception as e:
                # Closing the browser is a normal way to end a session and it
                # surfaces here as a TargetClosedError. Whatever the cause, the
                # recording so far is the entire point - never lose it to a
                # traceback.
                stop_reason = f"{e.__class__.__name__}: {e}"
                progress("")
                progress(
                    f"Browser session ended ({e.__class__.__name__}); saving what was captured."
                )

            manifest["pages_attached"] = len(attached)
            manifest["stop_reason"] = stop_reason
    except Exception as e:
        # A launch that fails must not leave a silent empty directory.
        manifest["stop_reason"] = f"launch failed: {e.__class__.__name__}: {e}"
        session.finalize(manifest)
        raise
    session.finalize(manifest)
    return session.out_dir


def new_session_dir(root: Path) -> Path:
    return root / datetime.now().strftime("%Y%m%d-%H%M%S")
