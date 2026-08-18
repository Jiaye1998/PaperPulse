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
    estimated_input_cost_per_million: float = float(
        os.getenv("PAPERPULSE_EST_INPUT_USD_PER_MILLION", "1.0")
    )
    estimated_output_cost_per_million: float = float(
        os.getenv("PAPERPULSE_EST_OUTPUT_USD_PER_MILLION", "6.0")
    )
    openalex_api_key: str = os.getenv("OPENALEX_API_KEY", "")
    # Crossref and OpenAlex move identified clients into their "polite" pool, which
    # is what keeps a few hundred lookups per refresh from being throttled to 429.
    contact_email: str = os.getenv("PAPERPULSE_CONTACT_EMAIL", "")
    # Elsevier deposits no abstracts to any open service, so its catalogue is only
    # reachable through Scopus, and only for an entitled institution. The optional
    # institution token extends that entitlement off the campus network.
    elsevier_api_key: str = os.getenv("ELSEVIER_API_KEY", "")
    elsevier_insttoken: str = os.getenv("ELSEVIER_INSTTOKEN", "")
    browser_abstracts: bool = os.getenv(
        "PAPERPULSE_BROWSER_ABSTRACTS", "true"
    ).lower() in {"1", "true", "yes", "on"}
    browser_headless: bool = os.getenv(
        "PAPERPULSE_BROWSER_HEADLESS", "true"
    ).lower() in {"1", "true", "yes", "on"}
    browser_timeout_ms: int = int(
        os.getenv("PAPERPULSE_BROWSER_TIMEOUT_MS", "12000")
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

    @property
    def browser_profile_dir(self) -> Path:
        return self.data_dir / "browser-profile"


config = AppConfig()
config.data_dir.mkdir(parents=True, exist_ok=True)
config.uploads_dir.mkdir(parents=True, exist_ok=True)
config.browser_profile_dir.mkdir(parents=True, exist_ok=True)
