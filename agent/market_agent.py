"""Deterministic analytical interpretation layer for Hebrew market briefings.

The component is deliberately rule-based: it reads computed pipeline artifacts
and turns them into a cautious research summary. It does not claim to predict
real-world events and it does not assign journalistic truth reliability.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

RELIABILITY_HE = {
    "good": "אמינות גבוהה",
    "medium": "אמינות בינונית",
    "weak": "אמינות נמוכה",
}

DIRECTION_HE = {
    "up": "עלייה",
    "down": "ירידה",
    "flat": "יציב",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _primary_model(ml_results: Dict[str, Any]) -> Dict[str, Any]:
    targets = ml_results.get("targets") or {}
    for horizon in ("1d", "1h"):
        result = targets.get(horizon) or {}
        if result.get("status") == "trained":
            return dict(result, horizon=horizon)
    return {}


def _source_labels(ml_results: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in (ml_results.get("source_labels") or {}).items()}


def _top_sources(articles: Sequence[Dict[str, Any]], ml_results: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    labels = _source_labels(ml_results)
    counts: Dict[str, int] = {}
    sentiments: Dict[str, List[float]] = {}
    for article in articles:
        source = str(article.get("source") or "Unknown")
        counts[source] = counts.get(source, 0) + 1
        if article.get("sentiment") is not None:
            sentiments.setdefault(source, []).append(_safe_float(article.get("sentiment")))

    scores: Dict[str, float] = {}
    primary = _primary_model(ml_results)
    for item in primary.get("source_importance") or []:
        slug = str(item.get("source"))
        source = labels.get(slug, slug.replace("_", " ").title())
        scores[source] = scores.get(source, 0.0) + _safe_float(item.get("importance"))

    all_sources = {source for source, score in scores.items() if score > 0}
    rows = []
    for source in all_sources:
        values = sentiments.get(source, [])
        n_articles = counts.get(source, 0)
        note = "מדד זה מייצג שימושיות חזויה בלבד, לא אמינות עיתונאית או אמת עובדתית."
        if n_articles < 5:
            note += " המדגם למקור זה קטן ולכן יש לפרש בזהירות."
        rows.append(
            {
                "source": source,
                "article_count": n_articles,
                "average_sentiment": round(sum(values) / len(values), 4) if values else 0.0,
                "predictive_score": round(scores.get(source, 0.0), 6),
                "note": note,
            }
        )
    return sorted(rows, key=lambda row: (row["predictive_score"], row["article_count"]), reverse=True)[:limit]


def _market_lookup(markets: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for market in markets:
        market_id = str(market.get("market_id") or market.get("question") or "")
        lookup[market_id] = market
    return lookup


def _top_markets(markets: Sequence[Dict[str, Any]], ml_results: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    primary = _primary_model(ml_results)
    market_by_id = _market_lookup(markets)
    rows: List[Dict[str, Any]] = []
    for pred in primary.get("predictions") or []:
        market_id = str(pred.get("market_id") or pred.get("market_question") or "")
        market = market_by_id.get(market_id, {})
        predicted_delta = _safe_float(pred.get("predicted_delta"))
        article_volume = int(pred.get("article_volume") or 0)
        direction = str(pred.get("predicted_direction") or "flat")
        if abs(predicted_delta) >= 0.005 and article_volume > 0:
            status = "למעקב"
        else:
            status = "אות חלש"
        explanation = (
            f"שינוי חזוי של {predicted_delta:+.3f}, "
            f"כיוון {DIRECTION_HE.get(direction, direction)}, וכיסוי של {article_volume} כתבות רלוונטיות."
        )
        rows.append(
            {
                "market_id": market_id,
                "market": pred.get("market_question") or market.get("question") or "",
                "current_probability": pred.get("current_probability"),
                "predicted_probability": pred.get("predicted_probability"),
                "predicted_delta": round(predicted_delta, 4),
                "direction": DIRECTION_HE.get(direction, direction),
                "article_volume": article_volume,
                "status": status,
                "why_selected": explanation,
                "url": market.get("url"),
            }
        )
    return sorted(rows, key=lambda row: abs(row["predicted_delta"]), reverse=True)[:limit]


def _risks(run: Dict[str, Any], ml_results: Dict[str, Any], articles: Sequence[Dict[str, Any]]) -> List[str]:
    risks: List[str] = []
    primary = _primary_model(ml_results)
    correlation = run.get("correlation") or {}
    p_value = correlation.get("p_value")
    if p_value is None or _safe_float(p_value, 1.0) >= 0.05:
        risks.append("לא נמצא מתאם Pearson מובהק סטטיסטית במדגם הנוכחי.")
    if primary and not primary.get("beats_baseline"):
        risks.append("המודל אינו משפר את תחזית הבסיס ולכן הערך החיזויי מוגבל.")
    if primary.get("reliability") == "weak":
        risks.append("תווית אמינות המודל נמוכה; התחזית מיועדת למחקר ולמעקב בלבד.")
    if len(articles) < 100:
        risks.append("מספר הכתבות קטן יחסית ועלול לגרום למדגם לא יציב.")
    risks.append("המודל חוזה תנועת הסתברות בפולימרקט, לא אמת עובדתית ולא תוצאה בעולם האמיתי.")
    return risks


def generate_agent_report(
    articles: Sequence[Dict[str, Any]],
    markets: Sequence[Dict[str, Any]],
    run: Dict[str, Any],
    ml_results: Dict[str, Any],
) -> Dict[str, Any]:
    primary = _primary_model(ml_results)
    reliability = primary.get("reliability")
    beats_baseline = bool(primary.get("beats_baseline"))
    top_markets = _top_markets(markets, ml_results)
    top_sources = _top_sources(articles, ml_results)
    risks = _risks(run, ml_results, articles)

    if reliability == "good" and beats_baseline:
        status = "בהרצה הנוכחית נמצא אות חיזוי חזק יחסית."
        next_action = "מומלץ לעקוב אחר השווקים הבולטים ולבדוק האם האות נשמר בהרצות נוספות."
    elif beats_baseline:
        status = "בהרצה הנוכחית נמצא אות חיזוי חלקי, אך יש לפרש אותו בזהירות."
        next_action = "מומלץ להמשיך לאסוף נתונים ולהשוות מול baseline בכל הרצה."
    else:
        status = (
            "בהרצה הנוכחית נמצא אות חיזוי חלש בלבד. המודל עדיין אינו משפר באופן עקבי "
            "את תחזית הבסיס, ולכן יש להתייחס לתחזיות כאינדיקציה מחקרית ראשונית ולא "
            "כמסקנה חיזויית חזקה."
        )
        next_action = "מומלץ להגדיל את המדגם ולהמשיך בתיקוף לפני הסקת מסקנות."

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_version": "rule_based_market_agent_v1",
        "status": status,
        "reliability": reliability,
        "reliability_he": RELIABILITY_HE.get(reliability, "לא זמין"),
        "beats_baseline": beats_baseline,
        "system_status": {
            "n_articles": len(articles),
            "n_markets": len(markets),
            "pearson_r": (run.get("correlation") or {}).get("pearson_r"),
            "pearson_p_value": (run.get("correlation") or {}).get("p_value"),
            "model_mae": primary.get("mae"),
            "baseline_mae": primary.get("baseline_mae"),
            "improvement_pct": primary.get("improvement_pct"),
            "directional_accuracy": primary.get("directional_accuracy"),
        },
        "top_sources": top_sources,
        "top_markets": top_markets,
        "risks_and_limitations": risks,
        "suggested_next_action": next_action,
        "forbidden_claims_note": (
            "שכבת הסיכום אינה טוענת שהאירועים יתרחשו במציאות, אינה מבטיחה תנועת שוק, "
            "ואינה מדרגת אמת או אמינות עיתונאית."
        ),
    }


def _read_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return default


def load_artifacts(output_dir: str = "output") -> Dict[str, Any]:
    return {
        "articles": _read_json(os.path.join(output_dir, "articles.json"), []),
        "markets": _read_json(os.path.join(output_dir, "markets.json"), []),
        "run": _read_json(os.path.join(output_dir, "run.json"), {}),
        "ml_results": _read_json(os.path.join(output_dir, "ml_results.json"), {}),
    }


def write_agent_report(report: Dict[str, Any], output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "agent_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return path


def generate_from_output(output_dir: str = "output") -> Dict[str, Any]:
    artifacts = load_artifacts(output_dir)
    report = generate_agent_report(
        artifacts["articles"],
        artifacts["markets"],
        artifacts["run"],
        artifacts["ml_results"] or (artifacts["run"].get("ml_forecast") or {}),
    )
    write_agent_report(report, output_dir)
    return report


if __name__ == "__main__":
    report = generate_from_output()
    print(json.dumps(report, ensure_ascii=False, indent=2))
