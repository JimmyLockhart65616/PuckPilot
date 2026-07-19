from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration. Values come from environment or the repo-root .env file.

    Relative paths are resolved against the repo root so scheduled tasks can run
    from any working directory.
    """

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    yahoo_client_id: str = ""
    yahoo_client_secret: str = ""
    yahoo_league_id: str = ""

    db_path: Path = Path("data/puckpilot.db")
    cache_dir: Path = Path("data/cache")
    token_path: Path = Path("secrets/oauth2.json")

    def _resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def resolved_db_path(self) -> Path:
        return self._resolve(self.db_path)

    @property
    def resolved_cache_dir(self) -> Path:
        return self._resolve(self.cache_dir)

    @property
    def resolved_token_path(self) -> Path:
        return self._resolve(self.token_path)
