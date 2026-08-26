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
Polymarket client and the news scraper to filter content, together with the
matching helpers below so every layer applies identical semantics.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Default geopolitical keyword set (Middle-East conflict focus).
# Kept module-level so it can be imported/extended without instantiating
# Settings, and used as the canonical filter across the whole pipeline.
# ---------------------------------------------------------------------------
# Every term must be *specific to the Middle East*. Generic topic words are
# deliberately excluded: a bare "ceasefire" pulls in "Russia x Ukraine ceasefire"
# and a bare "nuclear" pulls in "Will Russia test a nuclear weapon". The regional
# actors below already co-occur with those topics in the markets we want.
# "jordan" is likewise omitted — it matches "Jordan Bardella" and "Jordan Walker".
DEFAULT_GEO_KEYWORDS: List[str] = [
    # ── core actors ──────────────────────────────────────────────────────
    "israel",
    "israeli",
    "gaza",
    "west bank",
    "palestine",
    "palestinian",
    "lebanon",
    "hezbollah",
    "iran",
    "iranian",
    "hamas",
    "idf",
    "sinwar",
    "middle east",
    "netanyahu",
    "khamenei",
    "tehran",
    "jerusalem",
    # ── wider region ─────────────────────────────────────────────────────
    "syria",
    "yemen",
    "houthi",
    "saudi arabia",
    "egypt",
    "qatar",
    "turkey",
    "iraq",
    "uae",
    "emirates",
    "oman",
    "gulf cooperation council",
    # ── diplomacy & regional agreements ──────────────────────────────────
    "abraham accords",
    "knesset",
    "likud",
    # ── nuclear file (Iran-specific terms only) ──────────────────────────
    "jcpoa",
    "iaea",
    "fordow",
    "natanz",
    # ── economy & maritime ───────────────────────────────────────────────
    "hormuz",
    "opec",
    "kharg",
    # ── Hebrew (substring semantics — see keyword_pattern) ───────────────
    "נתניהו",
    "מלחמה",
    "הפסקת אש",
    "חיזבאללה",
    "חמאס",
    "איראן",
    "עזה",
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


# Negative keywords: reject a market even when a geopolitical term matches,
# because the event is sport/entertainment rather than a security question
# (e.g. "Will Iran win the 2026 FIFA World Cup?" matches "iran").
DEFAULT_EXCLUDED_MARKET_TERMS: Tuple[str, ...] = (
    # sport — "Will Iran win the 2026 FIFA World Cup?" matches "iran";
    # "Will Israel Adesanya be the UFC Middleweight Champion?" matches "israel".
    "fifa",
    "world cup",
    "football",
    "soccer",
    "sports",
    "sport",
    "olympics",
    "olympic",
    "ufc",
    "nba",
    "nfl",
    "mlb",
    "premier league",
    "champions league",
    "super bowl",
    "boxing",
    "grand slam",
    # awards — a market is not a geopolitical event just because a regional
    # figure is listed among the nominees in its description.
    "nobel peace prize",
    "nobel prize",
    "oscar",
    "grammy",
    "eurovision",
    # entertainment / celebrity
    "box office",
    "album",
    # Russia/Ukraine bilateral markets that merely name a Gulf state as the
    # summit venue ("Will Zelenskyy and Putin meet next in Qatar / UAE?").
    "zelenskyy",
    "zelensky",
    "putin",
)


# Coarse buckets used to spread the market basket across topics. Without this
# "iran" and "iranian" (or "israel" and "netanyahu") count as separate topics
# and the diversity pass happily fills the basket with the same subject twice.
KEYWORD_BUCKETS: Dict[str, str] = {
    "iran": "iran", "iranian": "iran", "tehran": "iran", "khamenei": "iran",
    "kharg": "iran", "jcpoa": "nuclear", "iaea": "nuclear", "fordow": "nuclear",
    "natanz": "nuclear",
    "israel": "israel", "israeli": "israel", "netanyahu": "israel",
    "jerusalem": "israel", "knesset": "israel", "likud": "israel", "idf": "israel",
    "נתניהו": "israel", "איראן": "iran", "עזה": "gaza",
    "gaza": "gaza", "west bank": "gaza", "palestine": "gaza",
    "palestinian": "gaza", "hamas": "gaza", "sinwar": "gaza", "חמאס": "gaza",
    "lebanon": "lebanon", "hezbollah": "lebanon", "חיזבאללה": "lebanon",
    "hormuz": "energy", "opec": "energy",
    "abraham accords": "diplomacy", "gulf cooperation council": "diplomacy",
    "הפסקת אש": "ceasefire", "מלחמה": "regional",
    "syria": "regional", "yemen": "regional", "houthi": "regional",
    "iraq": "regional", "egypt": "regional", "turkey": "regional",
    "middle east": "regional",
    "saudi arabia": "gulf", "qatar": "gulf", "uae": "gulf",
    "emirates": "gulf", "oman": "gulf",
}


def keyword_bucket(keyword: str) -> str:
    """Coarse topic bucket for ``keyword`` (falls back to the keyword itself)."""
    return KEYWORD_BUCKETS.get(keyword, keyword)


@lru_cache(maxsize=1024)
def keyword_pattern(keyword: str) -> Pattern[str]:
    """Compile ``keyword`` into a matcher with the project's agreed semantics.

    ASCII terms match case-insensitively **between word boundaries**, so "idf"
    no longer matches inside "midfielder" and "sport" no longer matches inside
    "transport".

    Hebrew terms deliberately keep substring semantics. Hebrew attaches its
    prefixes (ה/ו/ב/ל/כ/מ/ש) directly to the stem, so "מלחמה" must still match
    "המלחמה" — a leading ``\b`` would reject exactly the forms we want.
    """
    if keyword.isascii():
        return re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    return re.compile(re.escape(keyword))


def _haystack(texts: Iterable[Optional[str]]) -> str:
    return " ".join(str(t) for t in texts if t)


def matched_keywords(keywords: Iterable[str], *texts: Optional[str]) -> List[str]:
    """Return the subset of ``keywords`` that occurs in ``texts``."""
    haystack = _haystack(texts)
    if not haystack:
        return []
    return [kw for kw in keywords if keyword_pattern(kw).search(haystack)]


def matches_any(terms: Iterable[str], *texts: Optional[str]) -> bool:
    """True when any of ``terms`` occurs in ``texts`` (same matching rules)."""
    haystack = _haystack(texts)
    if not haystack:
        return False
    return any(keyword_pattern(term).search(haystack) for term in terms)


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

    firestore_articles_collection: str = "news_articles"
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
    # Spread the basket across topics instead of taking the top-N by volume,
    # which otherwise returns almost nothing but Iran leadership markets.
    market_diversify: bool = True
    market_topic_max_share: float = 0.40
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
