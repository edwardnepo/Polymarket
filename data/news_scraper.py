"""Multi-source news fetcher.

Pulls recent articles from several reputable News RSS feeds (international and
Israeli), keeps only those from the last ``LOOKBACK_DAYS`` days that match the
project's geopolitical keyword set, and (optionally) enriches them with the
article body text.

RSS is used as the primary, reputable, structured entry point.  The *same*
geopolitical keyword filter is applied to every outlet, so adding sources only
improves recall — relevance is still strictly enforced.  Full-text extraction
is best-effort and degrades gracefully to the RSS summary, so the downstream
sentiment model always has clean text to work with.

Usage::

    from data.news_scraper import NewsScraper

    scraper = NewsScraper()
    articles = scraper.fetch_articles(fetch_full_text=False)
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dateutil import parser as date_parser
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None  # type: ignore

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Matches the visible paragraph text inside article bodies.
_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)


def _clean_html(text: str) -> str:
    """Strip tags and collapse whitespace from an HTML fragment."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = (
        text.replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
    )
    return _WS_RE.sub(" ", text).strip()


class NewsScraper:
    """Fetches and filters recent articles from multiple reputable outlets."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.sources = self._resolve_sources(self.settings)
        self.keywords = self.settings.geo_keywords
        self.timeout = self.settings.http_timeout
        self._session = self._build_session(self.settings.http_max_retries)

    # ------------------------------------------------------------------ #
    # Source resolution
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_sources(settings: Settings) -> List[Tuple[str, str]]:
        """Flatten ``settings.news_feeds`` into ``(source_name, feed_url)`` pairs.

        De-duplicates URLs while preserving the first source label seen for each.
        """
        pairs: List[Tuple[str, str]] = []
        seen: set[str] = set()

        for source, urls in (settings.news_feeds or {}).items():
            for url in urls or []:
                if url and url not in seen:
                    seen.add(url)
                    pairs.append((source, url))

        return pairs

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; Polymarket-Geopolitics/1.0; "
                    "+research-pipeline)"
                ),
                "Accept": "application/rss+xml, application/xml, text/html;q=0.9",
            }
        )
        if Retry is not None:
            retry = Retry(
                total=max_retries,
                backoff_factor=0.6,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        return session

    # ------------------------------------------------------------------ #
    # Keyword filtering
    # ------------------------------------------------------------------ #
    def _matched_keywords(self, *texts: Optional[str]) -> List[str]:
        haystack = " ".join(t for t in texts if t)
        haystack_lower = haystack.lower()
        matched: List[str] = []
        for kw in self.keywords:
            if kw.isascii():
                if kw.lower() in haystack_lower:
                    matched.append(kw)
            elif kw in haystack:
                matched.append(kw)
        return matched

    # ------------------------------------------------------------------ #
    # RSS / Atom parsing
    # ------------------------------------------------------------------ #
    # Element local-names that may carry a publication date, in priority order.
    # Covers RSS 2.0 (<pubDate>), Dublin Core (<dc:date>) and Atom
    # (<published>/<updated>/<issued>), so a single robust path handles every
    # outlet regardless of its feed dialect.
    _DATE_TAGS: Tuple[str, ...] = (
        "pubDate",
        "published",
        "date",
        "updated",
        "issued",
        "created",
    )

    @staticmethod
    def _localname(tag: str) -> str:
        """Return the local part of an ElementTree tag (drops ``{namespace}``)."""
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    @classmethod
    def _parse_date(cls, raw: Optional[str]) -> Optional[datetime]:
        """Parse a date string from any RSS/Atom dialect into a UTC datetime.

        Uses ``dateutil`` (which understands RFC 822, ISO 8601 and most ad-hoc
        formats) and falls back to a fuzzy parse before giving up, so noisy or
        non-standard date strings are still recovered rather than discarded.
        """
        if not raw or not raw.strip():
            return None
        raw = raw.strip()
        dt: Optional[datetime] = None
        for fuzzy in (False, True):
            try:
                dt = date_parser.parse(raw, fuzzy=fuzzy)
                break
            except (ValueError, OverflowError, TypeError):
                dt = None
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @classmethod
    def _extract_date(
        cls,
        children: Dict[str, List[ET.Element]],
        fallback: Optional[datetime],
    ) -> Optional[datetime]:
        """Probe every known date tag on an entry; fall back when none parse.

        Returning ``fallback`` (the feed's channel date, else the fetch time)
        instead of ``None`` ensures the published date is never rendered as
        "Unknown" downstream.
        """
        for tag in cls._DATE_TAGS:
            for node in children.get(tag, []):
                parsed = cls._parse_date(node.text)
                if parsed is not None:
                    return parsed
        return fallback

    @classmethod
    def _extract_link(cls, children: Dict[str, List[ET.Element]]) -> str:
        """Resolve an item URL across RSS (<link> text) and Atom (<link href>)."""
        candidates = children.get("link", [])
        # Atom: prefer the canonical alternate link; fall back to any href.
        for node in candidates:
            rel = (node.attrib.get("rel") or "alternate").lower()
            href = (node.attrib.get("href") or "").strip()
            if href and rel == "alternate":
                return href
        for node in candidates:
            href = (node.attrib.get("href") or "").strip()
            if href:
                return href
        # RSS: the URL is the element's text content.
        for node in candidates:
            if node.text and node.text.strip():
                return node.text.strip()
        # Last resort: a permalink-style <guid>.
        for node in children.get("guid", []):
            text = (node.text or "").strip()
            if text.startswith("http"):
                return text
        return ""

    @classmethod
    def _children_by_localname(cls, node: ET.Element) -> Dict[str, List[ET.Element]]:
        """Group an element's direct children by their namespace-stripped name."""
        grouped: Dict[str, List[ET.Element]] = {}
        for child in list(node):
            grouped.setdefault(cls._localname(child.tag), []).append(child)
        return grouped

    def _parse_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("Failed to parse RSS XML: %s", exc)
            return items

        # Channel-level timestamp, used to backfill entries that omit a date.
        channel_date: Optional[datetime] = None
        for el in root.iter():
            if self._localname(el.tag) in ("lastBuildDate", "pubDate") and el.text:
                channel_date = self._parse_date(el.text)
                if channel_date is not None:
                    break
        # Final fallback so a date is *always* present: the moment we fetched it.
        fallback_date = channel_date or datetime.now(timezone.utc)

        # RSS 2.0 wraps articles in <item>; Atom uses <entry>. Support both.
        entries = [
            el for el in root.iter() if self._localname(el.tag) in ("item", "entry")
        ]
        for entry in entries:
            children = self._children_by_localname(entry)
            link = self._extract_link(children)
            title_nodes = children.get("title", [])
            title = _clean_html(title_nodes[0].text or "") if title_nodes else ""
            # RSS uses <description>; Atom uses <summary> or <content>.
            summary_node = next(
                (
                    children[k][0]
                    for k in ("description", "summary", "content")
                    if children.get(k)
                ),
                None,
            )
            summary = _clean_html(summary_node.text or "") if summary_node is not None else ""
            published = self._extract_date(children, fallback_date)
            if not link or not title:
                continue
            items.append(
                {
                    "title": title,
                    "url": link,
                    "summary": summary,
                    "published_at": published,
                }
            )
        return items

    # ------------------------------------------------------------------ #
    # Optional full-text extraction
    # ------------------------------------------------------------------ #
    def fetch_full_text(self, url: str, max_chars: int = 5000) -> str:
        """Best-effort extraction of an article body; empty string on failure."""
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Full-text fetch failed for %s: %s", url, exc)
            return ""

        paragraphs = [_clean_html(p) for p in _P_RE.findall(resp.text)]
        # Keep substantial paragraphs only (filters nav/boilerplate fragments).
        body = " ".join(p for p in paragraphs if len(p) > 40)
        return body[:max_chars]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_articles(
        self,
        lookback_days: Optional[int] = None,
        fetch_full_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return de-duplicated, keyword-matched articles from the last N days.

        Args:
            lookback_days: Recency window (defaults to ``settings.lookback_days``).
            fetch_full_text: When ``True``, also download each article body.
        """
        lookback_days = lookback_days or self.settings.lookback_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        seen_urls: set[str] = set()
        articles: List[Dict[str, Any]] = []
        per_source: Dict[str, int] = {}

        for source_name, feed_url in self.sources:
            try:
                resp = self._session.get(feed_url, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Failed to fetch feed %s (%s): %s", feed_url, source_name, exc)
                continue

            for entry in self._parse_feed(resp.text):
                url = entry["url"]
                if url in seen_urls:
                    continue

                published = entry["published_at"]
                if published is not None and published < cutoff:
                    continue  # too old

                matched = self._matched_keywords(entry["title"], entry["summary"])
                if not matched:
                    continue  # strict geopolitical keyword filter

                seen_urls.add(url)
                article = {
                    "article_id": hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
                    "title": entry["title"],
                    "url": url,
                    "summary": entry["summary"],
                    "source": source_name,
                    "source_feed": feed_url,
                    "published_at": published.isoformat() if published else None,
                    "matched_keywords": matched,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                if fetch_full_text:
                    article["body"] = self.fetch_full_text(url)
                articles.append(article)
                per_source[source_name] = per_source.get(source_name, 0) + 1

        articles.sort(key=lambda a: a.get("published_at") or "", reverse=True)
        logger.info(
            "Fetched %d unique geopolitical articles from %d sources (last %d days): %s",
            len(articles),
            len(per_source),
            lookback_days,
            per_source,
        )
        return articles

    # ------------------------------------------------------------------ #
    @staticmethod
    def analysis_text(article: Dict[str, Any]) -> str:
        """Best text to feed the sentiment model: title + body|summary."""
        parts = [article.get("title", "")]
        parts.append(article.get("body") or article.get("summary") or "")
        return " ".join(p for p in parts if p).strip()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "NewsScraper":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


if __name__ == "__main__":  # `python -m data.news_scraper`
    from config.logging_config import configure_logging

    configure_logging()
    _scraper = NewsScraper()
    _items = _scraper.fetch_articles()
    for _a in _items[:10]:
        print(f"[{_a['published_at']}] ({_a['source']}) {_a['title']}  -> {_a['matched_keywords']}")
