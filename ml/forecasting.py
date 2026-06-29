"""Forecasting dataset and explainable market-movement models.

This module is intentionally additive: it consumes the existing article and
market records, builds a source-aware supervised-learning dataset, then trains
small tree models that try to predict future Polymarket probability movement.
The current Pearson analysis remains the statistical baseline elsewhere.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)

TARGET_HORIZONS: Dict[str, int] = {"1h": 1, "1d": 24}
TARGET_COLUMNS: Dict[str, str] = {
    label: f"target_delta_{label}" for label in TARGET_HORIZONS
}
MOVEMENT_THRESHOLD = 0.005
MODEL_MIN_ROWS = 12
TRAIN_FRACTION = 0.80
ARTICLE_WINDOWS_HOURS: Tuple[int, ...] = (1, 3, 6, 12, 24)

TOPIC_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "iran": ("iran", "tehran", "khamenei", "irgc", "revolutionary guard"),
    "israel": ("israel", "israeli", "idf", "netanyahu", "jerusalem", "tel aviv", "נתניהו"),
    "gaza": ("gaza", "rafah", "khan younis", "gazans"),
    "hezbollah": ("hezbollah", "nasrallah"),
    "hamas": ("hamas", "sinwar"),
    "nuclear_deal": ("nuclear", "uranium", "jcpoa", "enrichment", "nuke"),
    "ceasefire": ("ceasefire", "truce", "hostage deal", "deal", "הפסקת אש"),
    "sanctions": ("sanction", "sanctions", "embargo"),
    "regional_war": ("regional war", "middle east war", "escalation", "strike", "missile", "attack", "war"),
}

_NON_FEATURE_COLUMNS = {
    "market_id",
    "market_question",
    "timestamp",
    "unix",
    "matched_keywords",
    "topic_labels",
    "future_probability_1h",
    "future_probability_1d",
    "target_delta_1h",
    "target_delta_1d",
    "target_direction_1h",
    "target_direction_1d",
}


def _coerce_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _slug(text: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")
    return value or "unknown"


def _topic_labels(*texts: Any) -> set[str]:
    haystack = " ".join(
        " ".join(str(v) for v in text) if isinstance(text, (list, tuple, set)) else str(text or "")
        for text in texts
    ).lower()
    labels: set[str] = set()
    for topic, terms in TOPIC_KEYWORDS.items():
        if any(term.lower() in haystack for term in terms):
            labels.add(topic)
    return labels


def _movement_label(delta: Optional[float]) -> Optional[str]:
    if delta is None or pd.isna(delta):
        return None
    if delta > MOVEMENT_THRESHOLD:
        return "up"
    if delta < -MOVEMENT_THRESHOLD:
        return "down"
    return "flat"


def _parse_articles(articles: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for article in articles:
        published = _coerce_dt(article.get("published_at"))
        if published is None:
            continue
        rows.append(
            {
                "article_id": article.get("article_id"),
                "published_at": published,
                "source": article.get("source") or "Unknown",
                "source_slug": _slug(article.get("source") or "Unknown"),
                "sentiment": float(article.get("sentiment") or 0.0),
                "matched_keywords": {
                    str(kw).lower() for kw in (article.get("matched_keywords") or [])
                },
                "topic_labels": _topic_labels(
                    article.get("title"),
                    article.get("summary"),
                    article.get("body"),
                    article.get("matched_keywords") or [],
                ),
            }
        )
    return pd.DataFrame(rows)


def _normalise_history(market: Dict[str, Any]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for point in market.get("price_history") or []:
        probability = point.get("probability")
        ts = point.get("unix")
        dt = _coerce_dt(point.get("timestamp"))
        if probability is None or dt is None:
            continue
        if ts is None:
            ts = int(dt.timestamp())
        try:
            points.append(
                {
                    "timestamp": dt,
                    "unix": int(ts),
                    "probability": float(probability),
                }
            )
        except (TypeError, ValueError):
            continue
    points.sort(key=lambda p: p["unix"])
    return points


def _future_probability(
    history: Sequence[Dict[str, Any]],
    index: int,
    horizon_hours: int,
) -> Optional[float]:
    target_unix = int(history[index]["unix"]) + horizon_hours * 3600
    for point in history[index + 1 :]:
        if int(point["unix"]) >= target_unix:
            return float(point["probability"])
    return None


def _source_universe(article_df: pd.DataFrame, max_sources: int) -> List[Tuple[str, str]]:
    if article_df.empty:
        return []
    counts = article_df["source"].value_counts().head(max_sources)
    return [(_slug(source), source) for source in counts.index.tolist()]


def _article_window(
    article_df: pd.DataFrame,
    timestamp: datetime,
    market_keywords: set[str],
    market_topics: set[str],
    window_hours: int,
) -> pd.DataFrame:
    if article_df.empty:
        return article_df
    start = timestamp - timedelta(hours=window_hours)
    mask = (article_df["published_at"] <= timestamp) & (article_df["published_at"] >= start)
    window = article_df.loc[mask].copy()
    if window.empty:
        return window
    keyword_mask = window["matched_keywords"].apply(lambda kws: bool(kws & market_keywords))
    if market_topics and "topic_labels" in window.columns:
        topic_mask = window["topic_labels"].apply(lambda topics: bool(topics & market_topics))
    else:
        topic_mask = pd.Series(False, index=window.index)
    if market_keywords and market_topics:
        keep = keyword_mask & topic_mask
    else:
        keep = keyword_mask | topic_mask
    return window.loc[keep].copy()


def _window_features(
    article_df: pd.DataFrame,
    timestamp: datetime,
    market_keywords: set[str],
    market_topics: set[str],
    sources: Sequence[Tuple[str, str]],
    window_hours: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    window = _article_window(article_df, timestamp, market_keywords, market_topics, window_hours)
    suffix = f"{window_hours}h"
    features: Dict[str, Any] = {
        f"article_count_{suffix}": int(len(window)),
        f"mean_sentiment_{suffix}": float(window["sentiment"].mean()) if not window.empty else 0.0,
        f"sentiment_min_{suffix}": float(window["sentiment"].min()) if not window.empty else 0.0,
        f"sentiment_max_{suffix}": float(window["sentiment"].max()) if not window.empty else 0.0,
    }
    for source_slug, source_name in sources:
        source_rows = window.loc[window["source"] == source_name] if not window.empty else window
        features[f"source__{source_slug}__count_{suffix}"] = int(len(source_rows))
        features[f"source__{source_slug}__sentiment_{suffix}"] = (
            float(source_rows["sentiment"].mean()) if not source_rows.empty else 0.0
        )
    return window, features


def _market_rows(
    market: Dict[str, Any],
    article_df: pd.DataFrame,
    sources: Sequence[Tuple[str, str]],
    article_window_hours: int,
) -> List[Dict[str, Any]]:
    history = _normalise_history(market)
    if not history:
        return []

    market_keywords = {str(kw).lower() for kw in (market.get("matched_keywords") or [])}
    market_topics = _topic_labels(
        market.get("question"),
        market.get("description"),
        market.get("slug"),
        market.get("matched_keywords") or [],
    )
    rows: List[Dict[str, Any]] = []
    for idx, point in enumerate(history):
        timestamp = point["timestamp"]
        probability = float(point["probability"])
        window, window_features = _window_features(
            article_df,
            timestamp,
            market_keywords,
            market_topics,
            sources,
            article_window_hours,
        )
        hour = timestamp.hour
        dow = timestamp.weekday()
        row: Dict[str, Any] = {
            "market_id": market.get("market_id"),
            "market_question": market.get("question") or "",
            "timestamp": timestamp.isoformat(),
            "unix": int(point["unix"]),
            "matched_keywords": ", ".join(sorted(market_keywords)),
            "topic_labels": ", ".join(sorted(market_topics)),
            "historical_probability": probability,
            "article_volume": int(len(window)),
            "mean_sentiment": float(window["sentiment"].mean()) if not window.empty else 0.0,
            "sentiment_min": float(window["sentiment"].min()) if not window.empty else 0.0,
            "sentiment_max": float(window["sentiment"].max()) if not window.empty else 0.0,
            "market_volume": float(market.get("volume") or 0.0),
            "market_liquidity": float(market.get("liquidity") or 0.0),
            "history_point_index": idx,
            "hour_utc": hour,
            "day_of_week": dow,
            "is_weekend": 1 if dow >= 5 else 0,
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "dow_sin": math.sin(2 * math.pi * dow / 7),
            "dow_cos": math.cos(2 * math.pi * dow / 7),
        }
        if idx > 0:
            row["probability_change_prev"] = probability - float(history[idx - 1]["probability"])
        else:
            row["probability_change_prev"] = 0.0

        row.update(window_features)
        for window_hours in ARTICLE_WINDOWS_HOURS:
            if window_hours == article_window_hours:
                continue
            _window, lag_features = _window_features(
                article_df,
                timestamp,
                market_keywords,
                market_topics,
                sources,
                window_hours,
            )
            row.update(lag_features)

        suffix = f"{article_window_hours}h"
        for source_slug, _source_name in sources:
            row[f"source__{source_slug}__count"] = row.get(f"source__{source_slug}__count_{suffix}", 0)
            row[f"source__{source_slug}__sentiment"] = row.get(f"source__{source_slug}__sentiment_{suffix}", 0.0)
        for topic in TOPIC_KEYWORDS:
            row[f"topic__{topic}"] = 1 if topic in market_topics else 0

        for label, hours in TARGET_HORIZONS.items():
            future = _future_probability(history, idx, hours)
            delta = None if future is None else future - probability
            row[f"future_probability_{label}"] = future
            row[f"target_delta_{label}"] = delta
            row[f"target_direction_{label}"] = _movement_label(delta)

        rows.append(row)
    return rows


def build_ml_dataset(
    articles: Sequence[Dict[str, Any]],
    markets: Sequence[Dict[str, Any]],
    article_window_hours: int = 24,
    max_sources: int = 20,
) -> pd.DataFrame:
    """Build a supervised forecasting dataset from existing pipeline records.

    Each row is a market/timestamp observation. Features include signed
    sentiment, outlet-specific article volume/sentiment, total article volume,
    historical market probability, recent probability movement and time fields.
    Targets are future probability deltas for the next hour and next day.
    """
    article_df = _parse_articles(articles)
    sources = _source_universe(article_df, max_sources=max_sources)

    rows: List[Dict[str, Any]] = []
    for market in markets:
        rows.extend(_market_rows(market, article_df, sources, article_window_hours))

    dataset = pd.DataFrame(rows)
    logger.info(
        "Built ML dataset with %d observations, %d markets and %d source feature groups.",
        len(dataset),
        dataset["market_id"].nunique() if not dataset.empty else 0,
        len(sources),
    )
    return dataset


def _feature_columns(dataset: pd.DataFrame) -> List[str]:
    features: List[str] = []
    for column in dataset.columns:
        if column in _NON_FEATURE_COLUMNS:
            continue
        if column.startswith("future_probability_") or column.startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            features.append(column)
    return features


def _source_importance(feature_importance: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    totals: Dict[str, float] = {}
    for item in feature_importance:
        feature = str(item["feature"])
        if not feature.startswith("source__"):
            continue
        parts = feature.split("__", 2)
        if len(parts) >= 3:
            totals[parts[1]] = totals.get(parts[1], 0.0) + float(item["importance"])
    return [
        {"source": source, "importance": round(value, 6)}
        for source, value in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        if value > 0
    ]


def _latest_observations(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return dataset
    idx = dataset.groupby("market_id")["unix"].idxmax()
    return dataset.loc[idx].copy().sort_values("unix", ascending=False)


def _direction_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    true_labels = [_movement_label(float(v)) for v in y_true]
    pred_labels = [_movement_label(float(v)) for v in y_pred]
    if not true_labels:
        return None
    correct = sum(1 for actual, pred in zip(true_labels, pred_labels) if actual == pred)
    return correct / len(true_labels)


def _reliability_label(improvement_pct: Optional[float], r2: Optional[float]) -> str:
    if improvement_pct is not None and improvement_pct > 2 and r2 is not None and r2 >= 0.05:
        return "good"
    if improvement_pct is not None and improvement_pct > 0:
        return "medium"
    return "weak"


def _baseline_summary(
    fit_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    target_col: str,
) -> Dict[str, Any]:
    from sklearn.metrics import mean_absolute_error

    avg_delta = float(fit_df[target_col].mean()) if not fit_df.empty else 0.0
    prev_change = pd.to_numeric(
        test_df.get("probability_change_prev", 0.0), errors="coerce"
    ).fillna(0.0)
    baseline_preds = {
        "no_change": np.zeros(len(y_test)),
        "previous_change": prev_change.to_numpy(dtype=float),
        "average_delta": np.full(len(y_test), avg_delta),
    }
    baselines: Dict[str, Dict[str, Any]] = {}
    for name, preds in baseline_preds.items():
        baselines[name] = {
            "mae": round(float(mean_absolute_error(y_test, preds)), 6),
            "directional_accuracy": (
                round(float(_direction_accuracy(y_test, preds)), 6) if len(y_test) else None
            ),
        }
    best_name, best = min(baselines.items(), key=lambda item: item[1]["mae"])
    return {
        "baselines": baselines,
        "best_baseline": best_name,
        "baseline_mae": best["mae"],
        "baseline_directional_accuracy": best["directional_accuracy"],
    }


def _train_one_target(
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    horizon_label: str,
    random_state: int,
) -> Dict[str, Any]:
    train_df = dataset.dropna(subset=[target_col]).copy()
    train_df = train_df.loc[train_df[target_col].apply(np.isfinite)]
    summary: Dict[str, Any] = {
        "target": target_col,
        "horizon": horizon_label,
        "status": "insufficient_data",
        "n_observations": int(len(train_df)),
        "feature_importance": [],
        "source_importance": [],
        "predictions": [],
    }
    if len(train_df) < MODEL_MIN_ROWS or not feature_cols:
        return summary

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError as exc:
        summary["status"] = "missing_dependency"
        summary["error"] = "Install scikit-learn to train forecasting models."
        logger.warning("Skipping ML forecast: %s", exc)
        return summary

    train_df = train_df.sort_values("unix").reset_index(drop=True)
    split_at = max(1, int(len(train_df) * TRAIN_FRACTION))
    if len(train_df) - split_at < 1:
        split_at = len(train_df)
    fit_df = train_df.iloc[:split_at].copy()
    test_df = train_df.iloc[split_at:].copy()

    if test_df.empty:
        fit_df = train_df.copy()
        test_df = train_df.copy()
        evaluation_mode = "in_sample"
    else:
        evaluation_mode = "time_holdout"

    X_train = fit_df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = fit_df[target_col].astype(float)
    X_test = test_df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = test_df[target_col].astype(float)
    baseline = _baseline_summary(fit_df, test_df, y_test, target_col)

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model_mae = float(mean_absolute_error(y_test, preds))
    r2_value = float(r2_score(y_test, preds)) if len(y_test) >= 2 else None
    baseline_mae = float(baseline["baseline_mae"])
    improvement_pct = (
        ((baseline_mae - model_mae) / baseline_mae) * 100 if baseline_mae > 0 else None
    )
    directional_accuracy = _direction_accuracy(y_test, preds)
    reliability = _reliability_label(improvement_pct, r2_value)

    importance = [
        {"feature": feature, "importance": round(float(score), 6)}
        for feature, score in sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if float(score) > 0
    ]

    latest = _latest_observations(dataset)
    prediction_rows: List[Dict[str, Any]] = []
    if not latest.empty:
        latest_X = latest[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        latest_preds = model.predict(latest_X)
        for (_, row), pred_delta in zip(latest.iterrows(), latest_preds):
            current = float(row["historical_probability"])
            prediction_rows.append(
                {
                    "market_id": row.get("market_id"),
                    "market_question": row.get("market_question"),
                    "timestamp": row.get("timestamp"),
                    "current_probability": round(current, 4),
                    "predicted_delta": round(float(pred_delta), 4),
                    "predicted_probability": round(float(np.clip(current + pred_delta, 0, 1)), 4),
                    "predicted_direction": _movement_label(float(pred_delta)),
                    "article_volume": int(row.get("article_volume") or 0),
                    "mean_sentiment": round(float(row.get("mean_sentiment") or 0.0), 4),
                }
            )

    summary.update(
        {
            "status": "trained",
            "evaluation_mode": evaluation_mode,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "mae": round(model_mae, 6),
            "r2": round(r2_value, 6) if r2_value is not None else None,
            "baselines": baseline["baselines"],
            "best_baseline": baseline["best_baseline"],
            "baseline_mae": baseline["baseline_mae"],
            "baseline_directional_accuracy": baseline["baseline_directional_accuracy"],
            "improvement_pct": round(float(improvement_pct), 3) if improvement_pct is not None else None,
            "beats_baseline": bool(improvement_pct is not None and improvement_pct > 0),
            "directional_accuracy": (
                round(float(directional_accuracy), 6) if directional_accuracy is not None else None
            ),
            "reliability": reliability,
            "feature_importance": importance[:25],
            "source_importance": _source_importance(importance),
            "predictions": prediction_rows[:25],
        }
    )
    return summary


def train_forecasting_models(
    dataset: pd.DataFrame,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train one explainable regressor per forecast horizon."""
    if dataset.empty:
        return {
            "status": "empty_dataset",
            "n_observations": 0,
            "n_markets": 0,
            "targets": {},
        }

    feature_cols = _feature_columns(dataset)
    targets = {
        label: _train_one_target(dataset, feature_cols, target_col, label, random_state)
        for label, target_col in TARGET_COLUMNS.items()
    }
    return {
        "status": "complete",
        "n_observations": int(len(dataset)),
        "n_markets": int(dataset["market_id"].nunique()),
        "feature_columns": list(feature_cols),
        "topics": sorted(TOPIC_KEYWORDS),
        "targets": targets,
    }


def _compact_dataset_records(dataset: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return a dashboard/demo-sized dataset snapshot.

    The model trains on the full in-memory frame above. For repository/demo
    submission we persist only the columns needed by the Streamlit source
    relationship view, keeping the JSON safely below common GitHub file limits.
    """
    if dataset.empty:
        return []
    keep = [
        "market_id",
        "market_question",
        "timestamp",
        "unix",
        "topic_labels",
        "historical_probability",
        "article_volume",
        "mean_sentiment",
        "target_delta_1h",
        "target_delta_1d",
    ]
    keep.extend(
        col
        for col in dataset.columns
        if re.match(r"^source__.+__(count|sentiment)$", col)
    )
    compact = dataset[[col for col in keep if col in dataset.columns]].copy()
    return compact.replace({np.nan: None}).to_dict(orient="records")


def build_ml_artifacts(
    articles: Sequence[Dict[str, Any]],
    markets: Sequence[Dict[str, Any]],
    article_window_hours: int = 24,
    max_sources: int = 20,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return JSON-serialisable dataset records and model results."""
    article_df = _parse_articles(articles)
    source_labels = {
        source_slug: source_name
        for source_slug, source_name in _source_universe(article_df, max_sources=max_sources)
    }
    dataset = build_ml_dataset(
        articles,
        markets,
        article_window_hours=article_window_hours,
        max_sources=max_sources,
    )
    results = train_forecasting_models(dataset, random_state=random_state)
    results["source_labels"] = source_labels
    results["article_window_hours"] = article_window_hours
    results["article_windows_hours"] = list(ARTICLE_WINDOWS_HOURS)
    results["movement_threshold"] = MOVEMENT_THRESHOLD
    results["matching_note"] = (
        "Articles are matched to markets by overlapping geopolitical keywords "
        "and derived topic labels. RSS feeds are recent-only; no historical "
        "article backfill is fabricated."
    )
    records = _compact_dataset_records(dataset)
    return records, results


if __name__ == "__main__":  # `python -m ml.forecasting`
    import json
    import os

    snapshot_dir = os.environ.get("SNAPSHOT_DIR", "output")
    with open(os.path.join(snapshot_dir, "articles.json"), "r", encoding="utf-8") as fh:
        _articles = json.load(fh)
    with open(os.path.join(snapshot_dir, "markets.json"), "r", encoding="utf-8") as fh:
        _markets = json.load(fh)
    _dataset, _results = build_ml_artifacts(_articles, _markets)
    print(json.dumps({k: v for k, v in _results.items() if k != "targets"}, indent=2))
    print(f"dataset_rows={len(_dataset)}")
