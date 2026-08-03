from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _resolve_data_dir() -> Path:
    configured = os.getenv("PAPERPULSE_DATA_DIR", "./data")
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = _resolve_data_dir()
    frontend_url: str = os.getenv("PAPERPULSE_FRONTEND_URL", "http://localhost:3000")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    analysis_model: str = os.getenv("PAPERPULSE_ANALYSIS_MODEL", "gpt-5.6-luna")
    embedding_model: str = os.getenv(
        "PAPERPULSE_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    inoreader_client_id: str = os.getenv("INOREADER_CLIENT_ID", "")
    inoreader_client_secret: str = os.getenv("INOREADER_CLIENT_SECRET", "")
    inoreader_redirect_uri: str = os.getenv(
        "INOREADER_REDIRECT_URI",
        "http://localhost:8000/api/inoreader/callback",
    )
    demo_mode: bool = os.getenv("PAPERPULSE_DEMO_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    @property
    def database_path(self) -> Path:
        return self.data_dir / "paperpulse.sqlite3"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


config = AppConfig()
config.data_dir.mkdir(parents=True, exist_ok=True)
config.uploads_dir.mkdir(parents=True, exist_ok=True)
