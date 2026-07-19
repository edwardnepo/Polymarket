"""End-to-end orchestration pipeline.

Stages
------
1. **Ingest News RSS**   — fetch recent, keyword-filtered articles.
2. **Run ML sentiment**  — score each article with a Hugging Face model.
3. **Ingest Polymarket** — fetch matching markets + probability history.
4. **Correlate**         — compute a daily sentiment ↔ probability correlation.
5. **Persist**           — upsert everything to Firestore (skipped on --dry-run).

A local JSON snapshot is always written to ``--output-dir`` so the pipeline can
run (and feed the dashboard) even without Firebase credentials.

Examples
--------
    # Full run, writing to Firestore:
    python -m pipeline.run_pipeline

    # Validate end-to-end without touching Firestore:
    python -m pipeline.run_pipeline --dry-run

    # Tweak scope:
    python -m pipeline.run_pipeline --lookback-days 14 --max-markets 10 --full-text
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from scipy import stats

from agent.market_agent import generate_agent_report
from config.logging_config import configure_logging, get_logger
from config.settings import get_settings
from data.news_scraper import NewsScraper
from data.polymarket_client import PolymarketClient
from ml.forecasting import build_ml_artifacts
from ml.sentiment_analyzer import SentimentAnalyzer

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def _daily_sentiment(articles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Mean signed sentiment and article count per calendar day (UTC)."""
    rows = [
        {"date": a["published_at"][:10], "sentiment": a.get("sentiment", 0.0)}
        for a in articles
        if a.get("published_at")
    ]
    if not rows:
        return pd.DataFrame(columns=["date", "sentiment", "article_count"])
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("date")
        .agg(sentiment=("sentiment", "mean"), article_count=("sentiment", "size"))
        .reset_index()
    )
    return grouped


def _daily_probability(markets: List[Dict[str, Any]]) -> pd.DataFrame:
    """Volume-agnostic mean Yes-probability per day across all markets."""
    rows: List[Dict[str, Any]] = []
    for market in markets:
        for point in market.get("price_history", []):
            ts = point.get("timestamp")
            if ts:
                rows.append({"date": ts[:10], "probability": point.get("probability")})
    if not rows:
        return pd.DataFrame(columns=["date", "probability"])
    df = pd.DataFrame(rows)
    return df.groupby("date").agg(probability=("probability", "mean")).reset_index()


def build_correlation_summary(
    articles: List[Dict[str, Any]],
    markets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pearson correlation between daily news sentiment and market probability."""
    sentiment_df = _daily_sentiment(articles)
    probability_df = _daily_probability(markets)

    summary: Dict[str, Any] = {
        "n_sentiment_days": int(len(sentiment_df)),
        "n_probability_days": int(len(probability_df)),
        "pearson_r": None,
        "p_value": None,
        "n_overlapping_days": 0,
    }

    if sentiment_df.empty or probability_df.empty:
        return summary

    merged = pd.merge(sentiment_df, probability_df, on="date", how="inner")
    summary["n_overlapping_days"] = int(len(merged))

    if len(merged) >= 3 and merged["sentiment"].nunique() > 1 and merged["probability"].nunique() > 1:
        r, p = stats.pearsonr(merged["sentiment"], merged["probability"])
        summary["pearson_r"] = round(float(r), 4)
        summary["p_value"] = round(float(p), 4)
    else:
        logger.info("Not enough overlapping/variable days for a correlation estimate.")

    return summary


# --------------------------------------------------------------------------- #
# Snapshot persistence
# --------------------------------------------------------------------------- #
def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


def write_local_snapshot(
    output_dir: str,
    articles: List[Dict[str, Any]],
    markets: List[Dict[str, Any]],
    run_metadata: Dict[str, Any],
    ml_dataset: Optional[List[Dict[str, Any]]] = None,
    ml_results: Optional[Dict[str, Any]] = None,
    agent_report: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a local JSON snapshot (used as a Firestore-free data source)."""
    _write_json(os.path.join(output_dir, "articles.json"), articles)
    _write_json(os.path.join(output_dir, "markets.json"), markets)
    _write_json(os.path.join(output_dir, "run.json"), run_metadata)
    if ml_dataset is not None:
        _write_json(os.path.join(output_dir, "ml_dataset.json"), ml_dataset)
    if ml_results is not None:
        _write_json(os.path.join(output_dir, "ml_results.json"), ml_results)
    if agent_report is not None:
        _write_json(os.path.join(output_dir, "agent_report.json"), agent_report)
    logger.info("Wrote local snapshot to '%s'.", os.path.abspath(output_dir))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    dry_run: bool = False,
    lookback_days: Optional[int] = None,
    max_markets: Optional[int] = None,
    full_text: bool = False,
    skip_articles: bool = False,
    skip_markets: bool = False,
    skip_ml: bool = False,
    output_dir: str = "output",
) -> Dict[str, Any]:
    """Execute the full pipeline and return the run-metadata dict."""
    settings = get_settings()
    lookback_days = lookback_days or settings.lookback_days
    max_markets = max_markets or settings.max_markets
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("run_%Y%m%dT%H%M%SZ")

    logger.info(
        "Pipeline start (run_id=%s, dry_run=%s, lookback=%d days, max_markets=%d).",
        run_id,
        dry_run,
        lookback_days,
        max_markets,
    )

    # --- Stage 1 & 2: multi-source news ingest + sentiment -----------------
    articles: List[Dict[str, Any]] = []
    if not skip_articles:
        with NewsScraper(settings) as scraper:
            articles = scraper.fetch_articles(
                lookback_days=lookback_days, fetch_full_text=full_text
            )
        if articles:
            analyzer = SentimentAnalyzer(settings=settings)
            analyzer.analyze_articles(articles)
    else:
        logger.info("Skipping news ingest (--skip-articles).")

    # --- Stage 3: Polymarket ingest ----------------------------------------
    markets: List[Dict[str, Any]] = []
    if not skip_markets:
        with PolymarketClient(settings) as client:
            markets = client.collect(max_markets=max_markets, lookback_days=lookback_days)
    else:
        logger.info("Skipping Polymarket ingest (--skip-markets).")

    # --- Stage 4: Correlation ----------------------------------------------
    correlation = build_correlation_summary(articles, markets)

    # --- Stage 4b: Forecasting ML layer ------------------------------------
    ml_dataset: Optional[List[Dict[str, Any]]] = None
    ml_results: Optional[Dict[str, Any]] = None
    if skip_ml:
        logger.info("Skipping forecasting ML layer (--skip-ml).")
    else:
        ml_dataset, ml_results = build_ml_artifacts(articles, markets)

    finished_at = datetime.now(timezone.utc)
    run_metadata: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        "dry_run": dry_run,
        "lookback_days": lookback_days,
        "max_markets": max_markets,
        "n_articles": len(articles),
        "n_markets": len(markets),
        "keywords": settings.geo_keywords,
        "correlation": correlation,
        "ml_forecast": ml_results,
    }
    agent_report: Optional[Dict[str, Any]] = None
    if ml_results:
        agent_report = generate_agent_report(articles, markets, run_metadata, ml_results)
        run_metadata["agent_report"] = agent_report

    # --- Always write a local snapshot -------------------------------------
    write_local_snapshot(output_dir, articles, markets, run_metadata, ml_dataset, ml_results, agent_report)

    # --- Stage 5: Firestore persistence ------------------------------------
    if dry_run:
        logger.info("Dry-run: skipping Firestore writes.")
        run_metadata["persisted"] = False
    elif not settings.has_firebase_credentials:
        logger.warning(
            "No Firebase credentials configured — skipping Firestore writes. "
            "Data is available in the local snapshot only."
        )
        run_metadata["persisted"] = False
    else:
        from data.firebase_client import FirestoreClient  # local import: heavy SDK

        fs = FirestoreClient(settings)
        n_articles = fs.save_articles(articles) if articles else 0
        n_markets = fs.save_markets(markets) if markets else 0
        fs.save_run(run_metadata)
        run_metadata["persisted"] = True
        run_metadata["persisted_articles"] = n_articles
        run_metadata["persisted_markets"] = n_markets

    logger.info(
        "Pipeline complete: %d articles, %d markets, pearson_r=%s (%.2fs).",
        run_metadata["n_articles"],
        run_metadata["n_markets"],
        correlation.get("pearson_r"),
        run_metadata["duration_seconds"],
    )
    return run_metadata


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Ingest news RSS + Polymarket, score sentiment, correlate, forecast, persist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every stage but DO NOT write to Firestore (local snapshot only).",
    )
    parser.add_argument("--lookback-days", type=int, default=None, help="Recency window in days.")
    parser.add_argument("--max-markets", type=int, default=None, help="Max markets to collect.")
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Download full BBC article bodies (slower, richer sentiment input).",
    )
    parser.add_argument("--skip-articles", action="store_true", help="Skip news ingest stage.")
    parser.add_argument("--skip-markets", action="store_true", help="Skip Polymarket ingest stage.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip forecasting dataset/model stage.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for the local JSON snapshot (default: ./output).",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR.")
    parser.add_argument(
        "--no-json-logs",
        action="store_true",
        help="Emit human-readable logs instead of JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    configure_logging(
        level=args.log_level,
        json_logs=False if args.no_json_logs else None,
        force=True,
    )
    try:
        run(
            dry_run=args.dry_run,
            lookback_days=args.lookback_days,
            max_markets=args.max_markets,
            full_text=args.full_text,
            skip_articles=args.skip_articles,
            skip_markets=args.skip_markets,
            skip_ml=args.skip_ml,
            output_dir=args.output_dir,
        )
    except Exception:  # noqa: BLE001 - top-level guard, log full traceback
        logger.exception("Pipeline failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
