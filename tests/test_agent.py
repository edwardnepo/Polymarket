from __future__ import annotations

import json

from agent.market_agent import generate_agent_report, generate_from_output


def _articles():
    return [
        {"source": "Ynet", "sentiment": -0.4},
        {"source": "Ynet", "sentiment": -0.2},
        {"source": "Globes", "sentiment": 0.3},
    ]


def _markets():
    return [
        {
            "market_id": "m1",
            "question": "Will regional tensions increase?",
            "current_probability": 0.42,
            "url": "https://polymarket.example/m1",
        }
    ]


def _ml_results(beats_baseline: bool = True):
    return {
        "source_labels": {"ynet": "Ynet", "globes": "Globes"},
        "targets": {
            "1d": {
                "status": "trained",
                "mae": 0.01,
                "baseline_mae": 0.02,
                "improvement_pct": 50.0 if beats_baseline else -10.0,
                "beats_baseline": beats_baseline,
                "directional_accuracy": 0.62,
                "reliability": "good" if beats_baseline else "weak",
                "source_importance": [
                    {"source": "ynet", "importance": 0.12},
                    {"source": "globes", "importance": 0.03},
                ],
                "predictions": [
                    {
                        "market_id": "m1",
                        "market_question": "Will regional tensions increase?",
                        "current_probability": 0.42,
                        "predicted_probability": 0.45,
                        "predicted_delta": 0.03,
                        "predicted_direction": "up",
                        "article_volume": 3,
                    }
                ],
            }
        },
    }


def test_agent_report_is_deterministic_and_cautious():
    report = generate_agent_report(
        _articles(),
        _markets(),
        {"correlation": {"pearson_r": 0.2, "p_value": 0.4}},
        _ml_results(),
    )

    assert report["agent_version"] == "rule_based_market_agent_v1"
    assert report["beats_baseline"] is True
    assert report["top_sources"][0]["source"] == "Ynet"
    assert report["top_markets"][0]["direction"] == "עלייה"
    assert "שכבת הסיכום אינה טוענת" in report["forbidden_claims_note"]
    assert any("Polymarket" in risk or "פולימרקט" in risk for risk in report["risks_and_limitations"])


def test_agent_warns_when_model_does_not_beat_baseline():
    report = generate_agent_report(
        _articles(),
        _markets(),
        {"correlation": {"pearson_r": 0.2, "p_value": 0.4}},
        _ml_results(beats_baseline=False),
    )

    assert report["beats_baseline"] is False
    assert report["reliability"] == "weak"
    assert "חלש" in report["status"]
    assert "תחזית הבסיס" in " ".join(report["risks_and_limitations"])


def test_generate_from_output_writes_agent_report(tmp_path):
    (tmp_path / "articles.json").write_text(json.dumps(_articles()), encoding="utf-8")
    (tmp_path / "markets.json").write_text(json.dumps(_markets()), encoding="utf-8")
    (tmp_path / "run.json").write_text(
        json.dumps({"correlation": {"pearson_r": 0.2, "p_value": 0.4}}),
        encoding="utf-8",
    )
    (tmp_path / "ml_results.json").write_text(json.dumps(_ml_results()), encoding="utf-8")

    report = generate_from_output(str(tmp_path))

    written = json.loads((tmp_path / "agent_report.json").read_text(encoding="utf-8"))
    assert written["agent_version"] == report["agent_version"]
    assert written["top_sources"][0]["source"] == "Ynet"
