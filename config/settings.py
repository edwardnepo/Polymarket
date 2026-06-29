"""Centralised, validated application configuration.

All runtime configuration is defined here as a single immutable Pydantic
``Settings`` object.  Values are read (in priority order) from:

1. Environment variables.
2. A local ``.env`` file (see ``.env.template``).
3. The defaults declared below.

Import the singleton everywhere via :func:`get_settings`::

    from config.settings import get_settings
    settings = get_settings()
    print(settings.lookback_days)

The geopolitical keyword list is the single source of truth used by both the
Polymarket client and the BBC scraper to filter content.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Default geopolitical keyword set (Middle-East conflict focus).
# Kept module-level so it can be imported/extended without instantiating
# Settings, and used as the canonical filter across the whole pipeline.
# ---------------------------------------------------------------------------
DEFAULT_GEO_KEYWORDS: List[str] = [
    "israel",
    "gaza",
    "lebanon",
    "hezbollah",
    "iran",
    "hamas",
    "idf",
    "sinwar",
    "middle east",
    "netanyahu",
    "נתניהו",
    "מלחמה",
]

# Multi-source reputable RSS feeds, keyed by the outlet that published them.
# The scraper applies the SAME geopolitical keyword filter to every source, so
# adding outlets only broadens perspective — relevance is still enforced
# downstream. Notes:
#   • Al Jazeera was removed (its endpoint triggers SSL/ISP blocks on some nets).
#   • Reuters is omitted (it discontinued its public RSS feeds).
DEFAULT_NEWS_FEEDS: Dict[str, List[str]] = {
    "BBC News": [
        "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/rss.xml",
    ],
    "CNN": [
        "http://rss.cnn.com/rss/edition_meast.rss",
    ],
    "The Times of Israel": [
        "https://www.timesofisrael.com/feed/",
    ],
    "The Jerusalem Post": [
        "https://www.jpost.com/rss/rssfeedsmiddleeastnews.aspx",
    ],
    "Fox News": [
        "https://moxie.foxnews.com/google-publisher/world.xml",
    ],
    "The New York Times": [
        "https://rss.nytimes.com/services/xml/rss/nyt/MiddleEast.xml",
    ],
    "Ynet": [
        "https://www.ynet.co.il/Integration/StoryRss2.xml",
        "https://www.ynet.co.il/Integration/StoryRss1854.xml",
    ],
    "N12": [
        "https://www.mako.co.il/rss/news-israel.xml",
        "https://www.mako.co.il/rss/news-military.xml",
    ],
    "Israel Hayom": [
        "https://www.israelhayom.co.il/rss.xml",
    ],
    "Maariv": [
        "https://www.maariv.co.il/Rss/RssFeedsMivzakim",
        "https://www.maariv.co.il/Rss/RssFeedsNews",
    ],
    "Calcalist": [
        "https://www.calcalist.co.il/GeneralRSS/0,16335,L-8,00.xml",
    ],
    "Globes": [
        "https://www.globes.co.il/webservice/rss/rssfeeder.asmx/FeederNode?iID=585",
    ],
    "The Marker": [
        "https://www.themarker.com/cmlink/1.144",
    ],
}


class Settings(BaseSettings):
    """Strongly-typed, environment-driven configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Firebase / Firestore ────────────────────────────────────────────
    google_application_credentials: Optional[str] = Field(
        default=None,
        description="Path to the Firebase service-account JSON key file.",
    )
    firebase_credentials_json: Optional[str] = Field(
        default=None,
        description="Inline service-account JSON (overrides the file path).",
    )
    firebase_project_id: Optional[str] = Field(default=None)

    firestore_articles_collection: str = "bbc_articles"
    firestore_markets_collection: str = "polymarket_markets"
    firestore_history_collection: str = "polymarket_price_history"
    firestore_runs_collection: str = "pipeline_runs"

    # ── Polymarket API ──────────────────────────────────────────────────
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"
    http_timeout: int = 30
    http_max_retries: int = 3

    # ── Pipeline ────────────────────────────────────────────────────────
    lookback_days: int = 90
    max_markets: int = 25
    price_interval: str = "1d"
    price_fidelity: int = 60

    # ── Machine Learning ────────────────────────────────────────────────
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_force_cpu: bool = False

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = True

    # ── Domain filters ──────────────────────────────────────────────────
    geo_keywords: List[str] = Field(default_factory=lambda: list(DEFAULT_GEO_KEYWORDS))
    # ``news_feeds`` maps an outlet name -> its RSS feed URLs. This is the single,
    # canonical multi-source config consumed by the scraper.
    news_feeds: Dict[str, List[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_NEWS_FEEDS.items()}
    )

    # -- validators ------------------------------------------------------
    @field_validator("price_interval")
    @classmethod
    def _validate_interval(cls, value: str) -> str:
        allowed = {"max", "all", "1m", "1w", "1d", "6h", "1h"}
        if value not in allowed:
            raise ValueError(
                f"price_interval must be one of {sorted(allowed)}, got '{value}'."
            )
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        value = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}.")
        return value

    @field_validator("geo_keywords")
    @classmethod
    def _normalise_keywords(cls, value: List[str]) -> List[str]:
        # Lower-case ASCII keywords for case-insensitive matching; keep
        # non-ASCII (e.g. Hebrew) terms untouched.
        return [kw.strip().lower() if kw.isascii() else kw.strip() for kw in value if kw.strip()]

    # -- convenience -----------------------------------------------------
    @property
    def has_firebase_credentials(self) -> bool:
        """True when enough configuration exists to initialise Firestore."""
        return bool(self.firebase_credentials_json or self.google_application_credentials)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()


# Eagerly expose a module-level instance for convenient imports.
settings: Settings = get_settings()
