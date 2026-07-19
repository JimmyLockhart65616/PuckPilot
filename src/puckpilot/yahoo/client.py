from __future__ import annotations

from typing import Any

NHL_GAME_CODE = "nhl"


class YahooClient:
    """All Yahoo Fantasy I/O goes through this boundary — nothing else in the
    codebase may import yahoo_fantasy_api directly. Keeps the integration
    swappable and gives one place to record fixtures, audit writes, and rate-limit.
    """

    def __init__(self, oauth: Any, league_id: str | None = None):
        import yahoo_fantasy_api as yfa  # lazy: heavy import, mocked in unit tests

        self._yfa = yfa
        self._oauth = oauth
        self._game = yfa.Game(oauth, NHL_GAME_CODE)
        self._league_id = league_id or None

    def league_ids(self, year: int | None = None) -> list[str]:
        return self._game.league_ids(year=year)

    def _league_key(self) -> str:
        """Resolve a bare league id (from the league URL) to a full Yahoo league
        key like '465.l.12345'. Full keys pass through unchanged."""
        if not self._league_id:
            ids = self._game.league_ids()
            if len(ids) == 1:
                return ids[0]
            raise ValueError(
                f"YAHOO_LEAGUE_ID not set and you have {len(ids)} leagues: {ids}. "
                "Set YAHOO_LEAGUE_ID in .env."
            )
        if ".l." in self._league_id:
            return self._league_id
        ids = self._game.league_ids()
        for lid in ids:
            if lid.endswith(f".l.{self._league_id}"):
                return lid
        raise ValueError(f"League {self._league_id!r} not found among your leagues: {ids}")

    def league(self):
        return self._game.to_league(self._league_key())

    def league_overview(self) -> dict[str, Any]:
        lg = self.league()
        settings = lg.settings()
        team_key = lg.team_key()
        roster = lg.to_team(team_key).roster()
        return {
            "league_key": self._league_key(),
            "name": settings.get("name"),
            "num_teams": settings.get("num_teams"),
            "scoring_type": settings.get("scoring_type"),
            "playoff_start_week": settings.get("playoff_start_week"),
            "stat_categories": lg.stat_categories(),
            "roster_positions": lg.positions(),
            "team_key": team_key,
            "roster": roster,
        }
