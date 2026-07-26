from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ml.forecasting import build_ml_artifacts, build_ml_dataset


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _articles(start: datetime):
    return [
        {
            "article_id": "a1",
            "source": "Ynet",
            "published_at": _iso(start + timedelta(hours=1)),
            "sentiment": -0.7,
            "matched_keywords": ["israel", "iran"],
        },
        {
            "article_id": "a2",
            "source": "Globes",
            "published_at": _iso(start + timedelta(hours=2)),
            "sentiment": 0.4,
            "matched_keywords": ["israel"],
        },
    ]


def _market(start: datetime, market_id: str = "m1"):
    history = []
    for hour in range(0, 30):
        dt = start + timedelta(hours=hour)
        history.append(
            {
                "timestamp": _iso(dt),
                "unix": int(dt.timestamp()),
                "probability": 0.40 + hour * 0.005 + (hour % 3) * 0.002,
            }
        )
    return {
        "market_id": market_id,
        "question": "Will Israel and Iran reach a ceasefire?",
        "matched_keywords": ["israel", "iran"],
        "volume": 1000,
        "liquidity": 500,
        "price_history": history,
    }


def test_build_ml_dataset_preserves_source_features_and_targets():
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    dataset = build_ml_dataset(_articles(start), [_market(start)], article_window_hours=24)

    assert not dataset.empty
    assert {"market_id", "timestamp", "historical_probability"}.issubset(dataset.columns)
    assert "source__ynet__count" in dataset.columns
    assert "source__globes__sentiment" in dataset.columns

    row_after_articles = dataset.loc[dataset["article_volume"] > 0].iloc[0]
    assert row_after_articles["market_id"] == "m1"
    assert row_after_articles["source__ynet__count"] >= 1
    assert row_after_articles["target_delta_1h"] > 0
    assert row_after_articles["target_delta_1d"] > 0


def test_build_ml_artifacts_trains_when_sklearn_available():
    pytest.importorskip("sklearn")
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    articles = _articles(start)
    markets = [_market(start, "m1"), _market(start, "m2")]

    records, results = build_ml_artifacts(articles, markets, article_window_hours=24)

    assert records
    assert results["n_observations"] == len(records)
    assert results["targets"]["1h"]["status"] == "trained"
    assert results["targets"]["1h"]["feature_importance"]
    assert results["targets"]["1h"]["evaluation_mode"] == "time_holdout"
    assert results["targets"]["1h"]["baseline_mae"] is not None
    assert results["targets"]["1h"]["directional_accuracy"] is not None
    assert "moving_subset" in results["targets"]["1h"]["validation_report"]
    assert results["targets"]["1h"]["reliability"] in {"good", "medium", "weak"}
    assert results["source_labels"]["ynet"] == "Ynet"
