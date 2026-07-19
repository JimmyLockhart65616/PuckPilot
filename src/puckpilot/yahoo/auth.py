from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from puckpilot.config import Settings

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
# must exactly match the redirect URI registered on the Yahoo app
REDIRECT_URI = "https://localhost:9000"

SETUP_HELP = """\
Yahoo credentials are not configured. One-time setup (~2 min):

  1. Create an app at https://developer.yahoo.com/apps/create/
       Application Type: Confidential Client (Web Application)
       Redirect URI:     https://localhost:8000  (never actually called)
       API Permissions:  Fantasy Sports -> Read/Write
  2. Copy .env.example to .env in the repo root
  3. Paste the Client ID / Client Secret into YAHOO_CLIENT_ID / YAHOO_CLIENT_SECRET

Then re-run this command; the first run opens a browser login and caches a
refresh token at secrets/oauth2.json (no further logins needed).
"""


class MissingYahooCredentials(RuntimeError):
    def __init__(self) -> None:
        super().__init__(SETUP_HELP)


def authorize_url(settings: Settings, redirect_uri: str = REDIRECT_URI) -> str:
    """URL the user opens in a browser to grant access.

    Yahoo redirects to {redirect_uri}?code=... — the page won't load (nothing
    listens on localhost:9000) but the code is in the address bar.
    """
    if not settings.yahoo_client_id:
        raise MissingYahooCredentials()
    params = {
        "client_id": settings.yahoo_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def extract_code(pasted: str) -> str:
    """Accept either the bare code or the full redirect URL the user pasted."""
    pasted = pasted.strip()
    if "code=" in pasted:
        query = urlparse(pasted).query or pasted.split("?", 1)[-1]
        return parse_qs(query)["code"][0]
    return pasted


def exchange_code(settings: Settings, code: str, redirect_uri: str = REDIRECT_URI) -> dict:
    """Trade the auth code for tokens; persist them yahoo_oauth-compatibly.

    The token file keeps yahoo_oauth.OAuth2(from_file=...) working unchanged,
    including its refresh flow, so get_oauth_session needs no changes.
    """
    resp = httpx.post(
        TOKEN_URL,
        auth=(settings.yahoo_client_id, settings.yahoo_client_secret),
        data={
            "grant_type": "authorization_code",
            "code": extract_code(code),
            "redirect_uri": redirect_uri,
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed: HTTP {resp.status_code}: {resp.text[:300]}")
    tok = resp.json()
    payload = {
        "consumer_key": settings.yahoo_client_id,
        "consumer_secret": settings.yahoo_client_secret,
        "access_token": tok["access_token"],
        "refresh_token": tok.get("refresh_token"),
        "token_type": tok.get("token_type", "bearer"),
        "token_time": time.time(),
        "guid": tok.get("xoauth_yahoo_guid"),
    }
    token_path = settings.resolved_token_path
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(payload))
    return payload


def get_oauth_session(settings: Settings):
    """Build an authenticated yahoo_oauth.OAuth2 session, refreshing the token if stale.

    First-ever call is interactive (browser login + paste verifier); after that the
    cached refresh token at settings.token_path keeps it unattended forever.
    """
    if not settings.yahoo_client_id or not settings.yahoo_client_secret:
        raise MissingYahooCredentials()

    # Imported lazily: yahoo_oauth logs at import time and unit tests never need it.
    from yahoo_oauth import OAuth2

    token_path = settings.resolved_token_path
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if not token_path.exists():
        token_path.write_text(
            json.dumps(
                {
                    "consumer_key": settings.yahoo_client_id,
                    "consumer_secret": settings.yahoo_client_secret,
                }
            )
        )

    oauth = OAuth2(None, None, from_file=str(token_path))
    if not oauth.token_is_valid():
        oauth.refresh_access_token()
    return oauth
