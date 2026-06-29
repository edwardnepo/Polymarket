"""Polymarket data client.

Fetches prediction-market metadata from the **Gamma API**
(``https://gamma-api.polymarket.com``) and historical probabilities from the
**CLOB API** (``https://clob.polymarket.com/prices-history``), keeping only the
markets that match the project's geopolitical keyword set.

A binary Polymarket market resolves to a pair of CLOB token ids (one for the
``Yes`` outcome, one for ``No``).  We treat the **Yes-token price** as the
market-implied probability of the event, since each share pays out ``$1`` on a
correct resolution (price ∈ [0, 1]).

Typical use::

    from data.polymarket_client import PolymarketClient

    with PolymarketClient() as client:
        markets = client.collect(max_markets=20, lookback_days=30)
        for m in markets:
            print(m["question"], m["current_probability"], len(m["price_history"]))
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 ships with requests; import location is stable.
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover - extremely defensive fallback
    Retry = None  # type: ignore

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger(__name__)

# CLOB ``/prices-history`` interval buckets (a duration ending "now"), ordered
# shortest → longest with their approximate span in days. The endpoint exposes
# two mutually exclusive query modes: an ``interval`` keyword OR an explicit
# ``startTs``/``endTs`` window. The window mode is capped server-side (empirically
# it 400s for ranges wider than ~2 weeks, regardless of ``fidelity``), whereas
# the ``interval`` mode reliably returns the full series — so we request by
# interval and trim to the desired window locally.
_INTERVAL_SPANS: List[tuple] = [("1d", 1), ("1w", 7), ("1m", 31), ("max", None)]


class PolymarketClient:
    """Thin, resilient HTTP client for Polymarket's public APIs."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.gamma_url = self.settings.polymarket_gamma_url.rstrip("/")
        self.clob_url = self.settings.polymarket_clob_url.rstrip("/")
        self.timeout = self.settings.http_timeout
        self.keywords = self.settings.geo_keywords
        self._session = self._build_session(self.settings.http_max_retries)

    # ------------------------------------------------------------------ #
    # Session / HTTP plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Polymarket-Geopolitics/1.0 (+research-pipeline)",
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

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET ``url`` and return decoded JSON, raising on non-2xx."""
        logger.debug("GET %s params=%s", url, params)
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    # Keyword filtering
    # ------------------------------------------------------------------ #
    # Negative keywords: even if a geopolitical term matches, reject the market
    # when any of these sports/entertainment terms appear (e.g. "Will Iran win
    # the 2026 FIFA World Cup?" matches "iran" but is not a security event).
    # Stems are used so "Olympics"/"Olympic" and "sport"/"sports" both match.
    _EXCLUDED_TERMS: tuple = (
        "fifa",
        "world cup",
        "football",
        "soccer",
        "sport",
        "olympic",
    )

    def _matched_keywords(self, *texts: Optional[str]) -> List[str]:
        """Return the subset of configured keywords found in ``texts``."""
        haystack_parts: List[str] = []
        for text in texts:
            if text:
                haystack_parts.append(text)
        haystack = " ".join(haystack_parts)
        haystack_lower = haystack.lower()
        matched: List[str] = []
        for kw in self.keywords:
            # ASCII keywords were lower-cased in settings; match case-insensitively.
            needle = kw.lower() if kw.isascii() else kw
            hay = haystack_lower if kw.isascii() else haystack
            if needle in hay:
                matched.append(kw)
        return matched

    def _has_excluded_topic(self, *texts: Optional[str]) -> bool:
        """True if any negative keyword appears — used to hard-reject the market."""
        haystack = " ".join(t for t in texts if t).lower()
        return any(term in haystack for term in self._EXCLUDED_TERMS)

    # ------------------------------------------------------------------ #
    # Gamma: markets
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_iso(value: Any) -> Optional[datetime]:
        """Parse an ISO-8601 timestamp (e.g. ``2026-06-30T00:00:00Z``) to UTC."""
        if not value or not isinstance(value, str):
            return None
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _is_active_market(
        self, market: Dict[str, Any], now: Optional[datetime] = None
    ) -> bool:
        """Strict liveness test — keep only tradeable, future-dated markets.

        A market is considered live only when it is flagged ``active``, is not
        ``closed``/resolved, and its resolution deadline has **not** already
        passed. This filters out resolved, closed and past-deadline markets so
        the analysis is based strictly on live future events.
        """
        now = now or datetime.now(timezone.utc)
        if not market.get("active", False):
            return False
        if market.get("closed", False):
            return False
        end = self._parse_iso(market.get("end_date"))
        if end is not None and end <= now:
            return False  # deadline already passed → effectively resolved/closed
        return True

    @staticmethod
    def _parse_json_field(raw: Any) -> List[Any]:
        """Gamma encodes list fields as JSON strings; decode defensively."""
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
                return value if isinstance(value, list) else [value]
            except (ValueError, TypeError):
                return []
        return []

    def _normalise_market(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a raw Gamma market into our normalised record.

        Returns ``None`` if the market cannot be traded (no order book / no
        token ids), since such markets have no usable price history.
        """
        token_ids = self._parse_json_field(raw.get("clobTokenIds"))
        outcomes = self._parse_json_field(raw.get("outcomes")) or ["Yes", "No"]
        prices_raw = self._parse_json_field(raw.get("outcomePrices"))

        if not token_ids:
            return None

        # Locate the "Yes" outcome; fall back to the first outcome/token.
        yes_index = 0
        for i, outcome in enumerate(outcomes):
            if isinstance(outcome, str) and outcome.strip().lower() == "yes":
                yes_index = i
                break

        yes_token_id = token_ids[yes_index] if yes_index < len(token_ids) else token_ids[0]

        current_probability: Optional[float] = None
        if yes_index < len(prices_raw):
            try:
                current_probability = float(prices_raw[yes_index])
            except (TypeError, ValueError):
                current_probability = None

        def _to_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        market_id = str(
            raw.get("conditionId") or raw.get("slug") or raw.get("id") or yes_token_id
        )

        return {
            "market_id": market_id,
            "gamma_id": str(raw.get("id")) if raw.get("id") is not None else None,
            "condition_id": raw.get("conditionId"),
            "question": raw.get("question") or raw.get("groupItemTitle") or "",
            "slug": raw.get("slug"),
            "description": raw.get("description") or "",
            "yes_token_id": str(yes_token_id),
            "token_ids": [str(t) for t in token_ids],
            "outcomes": outcomes,
            "current_probability": current_probability,
            "volume": _to_float(raw.get("volumeNum") if raw.get("volumeNum") is not None else raw.get("volume")),
            "volume_24hr": _to_float(raw.get("volume24hr")),
            "liquidity": _to_float(raw.get("liquidityNum") if raw.get("liquidityNum") is not None else raw.get("liquidity")),
            "active": bool(raw.get("active", False)),
            "closed": bool(raw.get("closed", False)),
            "enable_order_book": bool(raw.get("enableOrderBook", False)),
            "start_date": raw.get("startDate"),
            "end_date": raw.get("endDate"),
            "url": f"https://polymarket.com/market/{raw.get('slug')}" if raw.get("slug") else None,
        }

    def _fetch_markets_page(
        self,
        limit: int,
        offset: int,
        ordered: bool,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": "true",
            "closed": "false",
            "archived": "false",
        }
        if ordered:
            params["order"] = "volumeNum"
            params["ascending"] = "false"
        data = self._get(f"{self.gamma_url}/markets", params=params)
        if isinstance(data, dict):  # some deployments wrap results in {"data": [...]}
            data = data.get("data", [])
        return data if isinstance(data, list) else []

    def list_geopolitical_markets(
        self,
        max_markets: Optional[int] = None,
        scan_limit: int = 1000,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Scan active markets and keep those matching the geopolitical keywords.

        Args:
            max_markets: Stop once this many matching markets are collected.
            scan_limit: Hard cap on how many raw markets to inspect (bounds work).
            page_size: Markets requested per Gamma page.

        Returns:
            A list of normalised, keyword-matched market records sorted by
            descending trading volume.
        """
        max_markets = max_markets or self.settings.max_markets
        matched: List[Dict[str, Any]] = []
        offset = 0
        ordered = True
        now = datetime.now(timezone.utc)
        skipped_inactive = 0

        while offset < scan_limit:
            try:
                page = self._fetch_markets_page(page_size, offset, ordered)
            except requests.HTTPError as exc:
                # Server-side ordering can occasionally be rejected — retry unordered.
                if ordered:
                    logger.warning("Ordered market fetch failed (%s); retrying unordered.", exc)
                    ordered = False
                    continue
                logger.error("Failed to fetch markets at offset %s: %s", offset, exc)
                break

            if not page:
                break

            for raw in page:
                matched_kw = self._matched_keywords(
                    raw.get("question"),
                    raw.get("slug"),
                    raw.get("description"),
                    raw.get("groupItemTitle"),
                )
                if not matched_kw:
                    continue
                # Hard-reject sports/entertainment markets (FIFA, World Cup, etc.)
                # even when a geopolitical keyword incidentally matches.
                if self._has_excluded_topic(
                    raw.get("question"),
                    raw.get("slug"),
                    raw.get("description"),
                    raw.get("groupItemTitle"),
                ):
                    skipped_inactive += 1
                    continue
                normalised = self._normalise_market(raw)
                if normalised is None:
                    continue
                # Strict liveness gate: drop resolved/closed/past-deadline markets
                # so downstream analysis covers only live future events.
                if not self._is_active_market(normalised, now):
                    skipped_inactive += 1
                    continue
                normalised["matched_keywords"] = matched_kw
                matched.append(normalised)

            offset += page_size
            if len(matched) >= max_markets * 3:
                # Collected a healthy surplus; stop scanning early.
                break

        matched.sort(key=lambda m: (m.get("volume") or 0.0), reverse=True)
        result = matched[:max_markets]
        logger.info(
            "Selected %d active geopolitical markets (scanned up to offset %d; "
            "skipped %d closed/resolved/past-deadline).",
            len(result),
            offset,
            skipped_inactive,
        )
        return result

    # ------------------------------------------------------------------ #
    # CLOB: price history
    # ------------------------------------------------------------------ #
    @staticmethod
    def _interval_for_days(days: int) -> str:
        """Smallest CLOB interval bucket that covers ``days`` of history."""
        for name, span in _INTERVAL_SPANS:
            if span is None or days <= span:
                return name
        return "max"

    def _request_history(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call ``/prices-history`` and return the raw ``history`` list."""
        data = self._get(f"{self.clob_url}/prices-history", params=params)
        return data.get("history", []) if isinstance(data, dict) else []

    def get_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: Optional[str] = None,
        fidelity: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the Yes-token price series as ``[{"timestamp", "probability"}]``.

        A ``start_ts``/``end_ts`` window may be supplied; we satisfy it by
        requesting the smallest covering ``interval`` (the only mode the CLOB
        endpoint reliably serves for multi-week ranges — the raw startTs/endTs
        window 400s past ~2 weeks) and trimming the result locally. ``fidelity``
        controls bucket size in minutes.
        """
        token_id = str(token_id)
        fidelity = int(fidelity or self.settings.price_fidelity)

        # Pick the interval covering the requested window.
        if interval is None:
            if start_ts is not None and end_ts is not None:
                span_days = max(1, round((int(end_ts) - int(start_ts)) / 86400))
                interval = self._interval_for_days(span_days)
            else:
                interval = self.settings.price_interval

        raw: List[Dict[str, Any]] = []
        try:
            raw = self._request_history(
                {"market": token_id, "interval": interval, "fidelity": fidelity}
            )
        except requests.HTTPError as exc:
            logger.warning(
                "Interval price history failed for token %s (interval=%s): %s",
                token_id, interval, exc,
            )

        # Fallback: a bounded startTs/endTs window (valid for short ranges) if
        # the interval query returned nothing.
        if not raw and start_ts is not None and end_ts is not None:
            try:
                raw = self._request_history(
                    {
                        "market": token_id,
                        "startTs": int(start_ts),
                        "endTs": int(end_ts),
                        "fidelity": fidelity,
                    }
                )
            except requests.HTTPError as exc:
                logger.warning("Windowed price history failed for token %s: %s", token_id, exc)
                raw = []

        series: List[Dict[str, Any]] = []
        for point in raw:
            ts = point.get("t")
            price = point.get("p")
            if ts is None or price is None:
                continue
            ts = int(ts)
            # Trim to the requested window (the interval query can overshoot it).
            if start_ts is not None and ts < int(start_ts):
                continue
            if end_ts is not None and ts > int(end_ts):
                continue
            series.append(
                {
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "unix": ts,
                    "probability": float(price),
                }
            )
        return series

    # ------------------------------------------------------------------ #
    # High-level orchestration
    # ------------------------------------------------------------------ #
    def collect(
        self,
        max_markets: Optional[int] = None,
        lookback_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch matching markets and attach their recent probability history."""
        lookback_days = lookback_days or self.settings.lookback_days
        now = datetime.now(timezone.utc)
        start_ts = int((now - timedelta(days=lookback_days)).timestamp())
        end_ts = int(now.timestamp())

        markets = self.list_geopolitical_markets(max_markets=max_markets)
        enriched: List[Dict[str, Any]] = []
        for market in markets:
            history = self.get_price_history(
                market["yes_token_id"],
                start_ts=start_ts,
                end_ts=end_ts,
                fidelity=self.settings.price_fidelity,
            )
            market["price_history"] = history
            market["history_points"] = len(history)
            if history and market.get("current_probability") is None:
                market["current_probability"] = history[-1]["probability"]
            market["fetched_at"] = now.isoformat()
            market["lookback_days"] = lookback_days
            enriched.append(market)
            logger.debug(
                "Market '%s': %d price points.", market.get("question"), len(history)
            )

        logger.info("Collected %d markets with price history.", len(enriched))
        return enriched

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "PolymarketClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


if __name__ == "__main__":  # Manual smoke test: `python -m data.polymarket_client`
    from config.logging_config import configure_logging

    configure_logging()
    with PolymarketClient() as _client:
        _markets = _client.collect(max_markets=5)
        for _m in _markets:
            print(
                f"{_m['question'][:70]:<72} "
                f"p={_m.get('current_probability')} "
                f"vol={_m.get('volume')} "
                f"points={_m.get('history_points')}"
            )
