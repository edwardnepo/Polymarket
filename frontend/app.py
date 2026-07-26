"""לוח מחוונים אינטראקטיבי: חדשות גאופוליטיות מול שוקי ניבוי.

לוח מחוונים פורמלי ב-Streamlit שמטרתו לענות בפשטות על שאלה אחת: האם מצב הרוח של
החדשות עולה בקנה אחד עם מה ששוקי הניבוי מעריכים שיקרה?

הניתוח הוא *חתך-רוחב* (cross-sectional): כל שוק ניבוי מהווה תצפית אחת, ומשווים בין
ההסתברות המשתמעת הנוכחית שלו לבין הסנטימנט החדשותי הממוצע של הנושאים (מילות המפתח)
שהוא מסקר. גישה זו פועלת כבר בהרצת ה-pipeline הראשונה — ללא צורך בהיסטוריית מחירים.

הנתונים נקראים מ-Firestore כאשר קיימות הרשאות, ואחרת מתוך תמונת המצב המקומית
בפורמט JSON שנכתבת על ידי ``pipeline.run_pipeline``.

הרצה::

    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Streamlit Cloud executes this file from ``frontend/`` and does not always add
# the repository root to PYTHONPATH. Add it explicitly before project imports.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from agent.market_agent import generate_agent_report


def _load_streamlit_secrets_into_env() -> None:
    """Expose Streamlit Cloud secrets as env vars before Settings is created."""
    secret_keys = (
        "FIREBASE_CREDENTIALS_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "FIREBASE_PROJECT_ID",
    )
    try:
        secrets = st.secrets
        for key in secret_keys:
            if os.environ.get(key):
                continue
            if key in secrets:
                os.environ[key] = str(secrets[key])
                continue
            lower_key = key.lower()
            if lower_key in secrets:
                os.environ[key] = str(secrets[lower_key])
        if "firebase" in secrets:
            firebase = secrets["firebase"]
            mapping = {
                "credentials_json": "FIREBASE_CREDENTIALS_JSON",
                "project_id": "FIREBASE_PROJECT_ID",
            }
            for source_key, env_key in mapping.items():
                if not os.environ.get(env_key) and source_key in firebase:
                    os.environ[env_key] = str(firebase[source_key])
    except Exception:
        return


_load_streamlit_secrets_into_env()
from config.settings import get_settings

# --------------------------------------------------------------------------- #
# Page setup & theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="חדשות מול שוקי ניבוי",
    layout="wide",
    initial_sidebar_state="expanded",
)

SETTINGS = get_settings()
DEFAULT_SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "output")

# Palette
GREEN = "#2e9e5b"
RED = "#d6453d"
NEUTRAL = "#9aa0a6"
BLUE = "#2f6fed"
INK = "#1f2937"

# Display-only translations for values produced by the data layer (logic stays
# in the original language; only the on-screen label changes).
SOURCE_LABELS_HE = {
    "Firestore": "Firestore",
    "Firestore + local ML": "Firestore + תוצאות ML מקומיות",
    "Local snapshot": "מצב דמו — נתונים מקומיים",
    "Firestore (error)": "Firestore (שגיאה)",
    "Firestore (no creds)": "Firestore (ללא הרשאות)",
}
DATA_SOURCE_OPTIONS_HE = {
    "Auto": "אוטומטי",
    "Firestore": "Firestore",
    "Local snapshot": "תמונת מצב מקומית",
}


def inject_css() -> None:
    """עיצוב קל שהופך את הלוח לפורמלי ומיושר לימין (RTL)."""
    st.markdown(
        """
        <style>
          /* Native RTL for the whole app so Hebrew aligns right by default. */
          body, .stApp {direction: rtl; text-align: right;}
          .block-container {padding-top: 2.2rem; max-width: 1250px;}
          .block-container h1, .block-container h2, .block-container h3,
          .block-container [data-testid="stMarkdownContainer"],
          .block-container [data-testid="stCaptionContainer"],
          section[data-testid="stSidebar"] {direction: rtl; text-align: right;}
          /* Metric cards keep RTL only — no custom background/border, so they
             inherit Streamlit's theme and stay legible in light *and* dark mode. */
          div[data-testid="stMetric"] {direction: rtl; text-align: right;}
          .callout {direction: rtl; text-align: right; border-radius: 14px;
              padding: 18px 22px; margin: 4px 0 6px 0;
              border: 1px solid rgba(0,0,0,.06); line-height: 1.6;}
          .callout h3 {margin: 0 0 6px 0; font-size: 1.15rem;}
          .callout p {margin: 0; font-size: 1.02rem; color: #374151;}
          .howto {direction: rtl; text-align: right; background: #f5f8ff;
              border-right: 4px solid #2f6fed; border-radius: 8px;
              padding: 10px 14px; margin: 2px 0 10px 0;
              font-size: .95rem; color: #374151;}
          .pill {display:inline-block; padding:2px 10px; border-radius:999px;
              font-size:.8rem; font-weight:600; margin-right:6px;}
          .kpi-card {direction:rtl; text-align:right; border:1px solid rgba(128,128,128,.25);
              border-radius:8px; padding:14px 16px; min-height:92px; background:rgba(128,128,128,.05);}
          .kpi-label {font-size:.86rem; color:#6b7280; margin-bottom:8px;}
          .kpi-value {font-size:1.35rem; font-weight:700; color:#111827; line-height:1.25;
              white-space:normal; overflow-wrap:anywhere;}
          .status-line {direction:rtl; text-align:right; margin:14px 0 4px 0;
              padding:12px 14px; border-radius:10px; border:1px solid #d7dee8;
              background:#f8fafc; color:#1f2937; font-size:1.02rem; font-weight:600;}
          /* Theme-neutral footer (no solid fill) so it reads in dark mode too. */
          .methodology {direction: rtl; text-align: right; margin-top: 8px;
              padding-top: 12px; border-top: 1px solid rgba(128,128,128,.30);
              font-size: .86rem; line-height: 1.65; color: #6b7280;}
          .methodology h4 {margin: 0 0 4px 0; font-size: .95rem; color: #6b7280;}
          .methodology p {margin: 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _read_snapshot_json(snapshot_dir: str, name: str, default):
    path = os.path.join(snapshot_dir, name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return default


@st.cache_data(show_spinner=False)
def load_local_ml_artifacts(snapshot_dir: str) -> Tuple[Dict, List[Dict]]:
    """טעינת תוצאות ML מקומיות גם כאשר מקור הנתונים הראשי הוא Firestore."""
    run = _read_snapshot_json(snapshot_dir, "run.json", {})
    ml_results = _read_snapshot_json(snapshot_dir, "ml_results.json", {})
    ml_dataset = _read_snapshot_json(snapshot_dir, "ml_dataset.json", [])
    if ml_results:
        run = dict(run)
        run["ml_forecast"] = ml_results
        run["_ml_loaded_from"] = "ml_results.json"
    return run, ml_dataset


@st.cache_data(show_spinner=False)
def load_local_snapshot(snapshot_dir: str) -> Tuple[List[Dict], List[Dict], Dict, List[Dict], Dict]:
    """טעינת כתבות/שווקים/הרצה מתוך תמונת המצב המקומית בפורמט JSON."""

    articles = _read_snapshot_json(snapshot_dir, "articles.json", [])
    markets = _read_snapshot_json(snapshot_dir, "markets.json", [])
    run, ml_dataset = load_local_ml_artifacts(snapshot_dir)
    agent_report = _read_snapshot_json(snapshot_dir, "agent_report.json", {})
    return articles, markets, run, ml_dataset, agent_report


@st.cache_data(show_spinner=True, ttl=300)
def load_firestore() -> Tuple[List[Dict], List[Dict], Dict]:
    """טעינת כתבות/שווקים מ-Firestore (נשמר במטמון ל-5 דקות)."""
    from data.firebase_client import FirestoreClient

    client = FirestoreClient(SETTINGS)
    return client.get_articles(), client.get_markets(), {}


def load_data(source: str, snapshot_dir: str) -> Tuple[List[Dict], List[Dict], Dict, List[Dict], Dict, str]:
    """איתור מקור הנתונים; מחזיר (articles, markets, run, source_used)."""
    if source in ("Auto", "Firestore"):
        if SETTINGS.has_firebase_credentials:
            try:
                articles, markets, run = load_firestore()
                if articles or markets or source == "Firestore":
                    local_run, ml_dataset = load_local_ml_artifacts(snapshot_dir)
                    agent_report = _read_snapshot_json(snapshot_dir, "agent_report.json", {})
                    if not run.get("ml_forecast") and local_run.get("ml_forecast"):
                        run = dict(run)
                        run["ml_forecast"] = local_run["ml_forecast"]
                        run["_ml_loaded_from"] = local_run.get("_ml_loaded_from")
                        run["finished_at"] = run.get("finished_at") or local_run.get("finished_at")
                        return articles, markets, run, ml_dataset, agent_report, "Firestore + local ML"
                    return articles, markets, run, ml_dataset, agent_report, "Firestore"
            except Exception as exc:  # noqa: BLE001
                if source == "Firestore":
                    st.error(f"טעינה מ-Firestore נכשלה: {exc}")
                    return [], [], {}, [], {}, "Firestore (error)"
                st.warning(f"Firestore אינו זמין, נעשה שימוש בתמונת המצב המקומית ({exc}).")
        elif source == "Firestore":
            st.error("לא הוגדרו הרשאות Firebase (ראו .env.template).")
            return [], [], {}, [], {}, "Firestore (no creds)"

    articles, markets, run, ml_dataset, agent_report = load_local_snapshot(snapshot_dir)
    return articles, markets, run, ml_dataset, agent_report, "Local snapshot"


def _coerce_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp string into an aware UTC datetime (or None)."""
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


def latest_data_timestamp(
    articles: List[Dict], markets: List[Dict], run: Dict
) -> Optional[datetime]:
    """Most recent timestamp across the loaded data — the 'last updated' moment."""
    best: Optional[datetime] = None
    for value in (
        *(a.get("fetched_at") for a in articles),
        *(a.get("published_at") for a in articles),
        *(m.get("fetched_at") for m in markets),
        run.get("finished_at"),
    ):
        dt = _coerce_dt(value)
        if dt is not None and (best is None or dt > best):
            best = dt
    return best


# Sports/entertainment terms that should never appear in a security dashboard.
# Mirrors the data-layer reject list so a stale snapshot/Firestore record that
# predates the upstream filter still can't surface (e.g. a FIFA World Cup market).
EXCLUDED_MARKET_TERMS = ("fifa", "world cup", "football", "soccer", "sport", "olympic")


def drop_excluded_topics(markets: List[Dict]) -> List[Dict]:
    """Drop markets whose text matches a sports/entertainment negative keyword."""
    def _excluded(m: Dict) -> bool:
        hay = " ".join(
            str(m.get(k) or "") for k in ("question", "slug", "description")
        ).lower()
        return any(term in hay for term in EXCLUDED_MARKET_TERMS)

    return [m for m in markets if not _excluded(m)]


def drop_past_markets(markets: List[Dict]) -> List[Dict]:
    """Drop any market whose ``end_date`` has already passed.

    A defensive, app-side guard (independent of the data layer) so that even a
    stale snapshot or Firestore record can never surface resolved/past-deadline
    events. Markets without a parseable end date are kept (they passed the
    upstream active filter and have no deadline to be 'past').
    """
    now = datetime.now(timezone.utc)
    live: List[Dict] = []
    for m in markets:
        end = _coerce_dt(m.get("end_date"))
        if end is not None and end <= now:
            continue
        live.append(m)
    return live


def apply_keyword_filter(
    articles: List[Dict], markets: List[Dict], keywords: List[str]
) -> Tuple[List[Dict], List[Dict]]:
    if not keywords:
        return articles, markets
    kw = set(keywords)

    def _matches(item: Dict) -> bool:
        return bool(set(item.get("matched_keywords", [])) & kw)

    return [a for a in articles if _matches(a)], [m for m in markets if _matches(m)]


# --------------------------------------------------------------------------- #
# Cross-sectional analysis (one observation per market)
# --------------------------------------------------------------------------- #
CROSS_SECTION_COLUMNS = [
    "question",
    "keywords",
    "sentiment",
    "probability",
    "article_count",
    "volume",
]


def average_sentiment(articles: List[Dict]) -> float:
    vals = [a.get("sentiment") for a in articles if a.get("sentiment") is not None]
    return float(np.mean(vals)) if vals else 0.0


def average_probability(markets: List[Dict]) -> float:
    vals = [m.get("current_probability") for m in markets if m.get("current_probability") is not None]
    return float(np.mean(vals)) if vals else 0.0


def keyword_sentiment_map(articles: List[Dict]) -> Dict[str, Tuple[float, int]]:
    """מיפוי כל מילת מפתח -> (סנטימנט ממוצע של הכתבות, מספר הכתבות)."""
    buckets: Dict[str, List[float]] = {}
    for a in articles:
        sent = a.get("sentiment")
        if sent is None:
            continue
        for kw in a.get("matched_keywords", []) or []:
            buckets.setdefault(kw.lower(), []).append(float(sent))
    return {kw: (float(np.mean(v)), len(v)) for kw, v in buckets.items() if v}


def cross_sectional_frame(articles: List[Dict], markets: List[Dict]) -> pd.DataFrame:
    """שורה אחת לכל שוק: ההסתברות הנוכחית מול סנטימנט מילות המפתח הקשורות."""
    kw_map = keyword_sentiment_map(articles)
    overall = average_sentiment(articles)

    rows: List[Dict[str, Any]] = []
    for m in markets:
        prob = m.get("current_probability")
        if prob is None:
            continue
        mkws = [k.lower() for k in (m.get("matched_keywords", []) or [])]
        pairs = [kw_map[k] for k in mkws if k in kw_map]
        if pairs:
            weighted = sum(s * c for s, c in pairs)
            count = sum(c for _, c in pairs)
            sentiment = weighted / count if count else overall
            article_count = count
        else:
            sentiment, article_count = overall, 0
        rows.append(
            {
                "question": (m.get("question") or "")[:70],
                "keywords": ", ".join(mkws),
                "sentiment": sentiment,
                "probability": float(prob),
                "article_count": article_count,
                "volume": float(m.get("volume") or 0.0),
            }
        )
    return pd.DataFrame(rows, columns=CROSS_SECTION_COLUMNS)


def cross_sectional_pearson(cs: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if cs.empty or not {"sentiment", "probability"}.issubset(cs.columns):
        return None, None
    sub = cs.dropna(subset=["sentiment", "probability"])
    if len(sub) >= 3 and sub["sentiment"].nunique() > 1 and sub["probability"].nunique() > 1:
        r, p = stats.pearsonr(sub["sentiment"], sub["probability"])
        return float(r), float(p)
    return None, None


# --------------------------------------------------------------------------- #
# פרשנות בשפה פשוטה (עברית)
# --------------------------------------------------------------------------- #
def describe_correlation(r: Optional[float], p: Optional[float], n: int) -> Tuple[str, str, str]:
    """מחזיר (כותרת, משפט הסבר בעברית, גוון) עבור ערך מתאם פירסון.

    ``tone`` הוא "good"/"bad"/"neutral" וקובע את צבע התיבה.
    """
    if r is None:
        return (
            "לא ניתן עדיין לחשב מתאם",
            "אין עדיין מספיק שווקים בעלי סיקור חדשותי תואם כדי לחשב מתאם מהימן. "
            "יש לאסוף נתונים נוספים ולהריץ מחדש את ה-pipeline, והערך יתעדכן אוטומטית.",
            "neutral",
        )

    mag = abs(r)
    if mag < 0.10:
        strength = "זניח"
    elif mag < 0.30:
        strength = "חלש"
    elif mag < 0.50:
        strength = "בינוני"
    elif mag < 0.70:
        strength = "חזק"
    elif mag < 0.90:
        strength = "חזק מאוד"
    else:
        strength = "כמעט מושלם"

    if r > 0:
        direction_adj = "חיובי"
        mechanism = "ככל שהסיקור נוטה יותר לעבר רגיעה ודיפלומטיה, ההסתברות שהשוק מייחס לאירוע נוטה לעלות"
    elif r < 0:
        direction_adj = "שלילי"
        mechanism = "ככל שהסיקור משקף יותר הסלמה ולחימה, ההסתברות שהשוק מייחס לאירוע נוטה לעלות"
    else:
        direction_adj = "ניטרלי"
        mechanism = "השניים נעים באופן בלתי תלוי זה בזה"

    headline = f"קשר {strength} ו{direction_adj}"
    sentence = (
        f"מתאם של {r:+.2f} מצביע על קשר <b>{strength} ו{direction_adj}</b> "
        f"בין {n} השווקים שנותחו: {mechanism}."
    )
    if p is not None:
        if p < 0.05:
            sentence += (
                f" תוצאה זו <b>מובהקת סטטיסטית</b> (p = {p:.3f}), "
                "ולכן אין זה סביר שהיא נובעת מהמקרה בלבד."
            )
        else:
            sentence += (
                f" עם זאת, התוצאה <b>אינה מובהקת סטטיסטית</b> (p = {p:.3f}), "
                "ולכן יש להתייחס אליה כרמז ולא כמסקנה חד-משמעית."
            )

    tone = "neutral" if mag < 0.10 or (p is not None and p >= 0.05) else ("good" if r > 0 else "bad")
    return headline, sentence, tone


def mood_label(avg_sent: float) -> str:
    """תיוג מילולי של מדד המתיחות הביטחונית (ערך גבוה = רגיעה, נמוך = הסלמה)."""
    if avg_sent > 0.33:
        return "רגיעה ודיפלומטיה"
    if avg_sent > 0.05:
        return "נטייה לרגיעה"
    if avg_sent < -0.33:
        return "הסלמה ולחימה"
    if avg_sent < -0.05:
        return "נטייה להסלמה"
    return "מתיחות מעורבת / ניטרלית"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_sentiment_gauge(avg_sent: float) -> go.Figure:
    """מד מהירות בגוונים מאדום (הסלמה) לירוק (רגיעה) עבור מדד המתיחות הממוצע (−1 עד +1)."""
    bar_color = GREEN if avg_sent > 0.05 else (RED if avg_sent < -0.05 else NEUTRAL)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(avg_sent, 3),
            number={"font": {"size": 44, "color": INK}, "valueformat": "+.2f"},
            gauge={
                "axis": {
                    "range": [-1, 1],
                    "tickvals": [-1, -0.5, 0, 0.5, 1],
                    "ticktext": ["−1<br>הסלמה ולחימה", "−0.5", "0", "+0.5", "+1<br>רגיעה ודיפלומטיה"],
                    "tickfont": {"size": 11},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [-1.0, -0.6], "color": "#e7674f"},
                    {"range": [-0.6, -0.2], "color": "#f3a48f"},
                    {"range": [-0.2, 0.2], "color": "#eceff1"},
                    {"range": [0.2, 0.6], "color": "#a8d5b6"},
                    {"range": [0.6, 1.0], "color": "#5db77e"},
                ],
                "threshold": {
                    "line": {"color": INK, "width": 4},
                    "thickness": 0.8,
                    "value": round(avg_sent, 3),
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def chart_scatter(cs: pd.DataFrame, r: Optional[float], p: Optional[float]) -> go.Figure:
    sub = cs.dropna(subset=["sentiment", "probability"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["sentiment"],
            y=sub["probability"],
            mode="markers",
            marker=dict(
                size=13,
                color=sub["probability"],
                colorscale="RdYlGn",
                cmin=0, cmax=1,
                showscale=False,
                line=dict(width=1.2, color="#33373d"),
            ),
            text=sub["question"],
            customdata=sub["keywords"],
            hovertemplate=(
                "<b>%{text}</b><br>נושא: %{customdata}"
                "<br>מדד מתיחות: %{x:+.2f}<br>הסתברות השוק: %{y:.0%}<extra></extra>"
            ),
            name="שוק",
        )
    )
    if len(sub) >= 2 and sub["sentiment"].nunique() > 1:
        slope, intercept = np.polyfit(sub["sentiment"], sub["probability"], 1)
        xs = np.linspace(sub["sentiment"].min(), sub["sentiment"].max(), 50)
        fig.add_trace(
            go.Scatter(
                x=xs, y=slope * xs + intercept, mode="lines",
                line=dict(color="#555", dash="dash", width=2),
                name="מגמה כללית",
                hoverinfo="skip",
            )
        )
    subtitle = f"מתאם פירסון r = {r:+.2f}" if r is not None else "אין עדיין מספיק נתונים למגמה"
    fig.update_layout(
        title=dict(text=f"כל נקודה מייצגת שוק אחד &nbsp;•&nbsp; {subtitle}", font=dict(size=15)),
        xaxis_title="מדד המתיחות בסיקור — הסלמה (משמאל) עד רגיעה (מימין)",
        yaxis_title="הסתברות השוק (כן)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=440,
        showlegend=False,
        plot_bgcolor="#fbfcfe",
    )
    fig.update_xaxes(range=[-1, 1], zeroline=True, zerolinecolor="#d0d4da", gridcolor="#eef1f5")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor="#eef1f5")
    return fig


def chart_heatmap(cs: pd.DataFrame) -> Optional[go.Figure]:
    labels = {
        "sentiment": "מתיחות בסיקור",
        "probability": "הסתברות השוק",
        "article_count": "נפח חדשות",
        "volume": "נפח מסחר",
    }
    cols = [c for c in labels if c in cs.columns]
    sub = cs[cols].dropna()
    sub = sub.loc[:, sub.nunique() > 1]   # drop constant columns
    if sub.shape[0] < 3 or sub.shape[1] < 2:
        return None
    corr = sub.corr()
    names = [labels[c] for c in corr.columns]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=names, y=names,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont={"size": 13},
            colorbar=dict(title="r", tickvals=[-1, 0, 1],
                          ticktext=["−1 (הפוך)", "0", "+1 (ישר)"]),
            hovertemplate="%{y} מול %{x}: r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def chart_market_bar(markets: List[Dict]) -> Optional[go.Figure]:
    rows = [
        {
            # Keep the full question — truncation hid the meaningful part of the
            # label (only the trailing "...June 7?" was visible before).
            "question": m.get("question") or "",
            "probability": m.get("current_probability"),
        }
        for m in markets
        if m.get("current_probability") is not None
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("probability").tail(15)
    fig = px.bar(
        df, x="probability", y="question", orientation="h",
        color="probability", color_continuous_scale="RdYlGn", range_color=[0, 1],
    )
    fig.update_layout(
        title=dict(text="מה השווקים מעריכים כעת", font=dict(size=15)),
        xaxis_title="הסתברות משתמעת ל'כן'", yaxis_title="",
        # Wide left margin + automargin so long market questions render in full.
        coloraxis_showscale=False, height=460, margin=dict(l=400, r=10, t=50, b=10),
        plot_bgcolor="#fbfcfe",
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(automargin=True)
    return fig


def _display_feature_name(feature: str) -> str:
    labels = {
        "historical_probability": "הסתברות היסטורית",
        "article_volume": "נפח כתבות",
        "mean_sentiment": "סנטימנט ממוצע",
        "sentiment_min": "סנטימנט מינימלי",
        "sentiment_max": "סנטימנט מקסימלי",
        "market_volume": "נפח מסחר",
        "market_liquidity": "נזילות",
        "probability_change_prev": "שינוי הסתברות קודם",
        "hour_utc": "שעה",
        "day_of_week": "יום בשבוע",
        "is_weekend": "סוף שבוע",
        "hour_sin": "שעה (מחזורי)",
        "hour_cos": "שעה (מחזורי)",
        "dow_sin": "יום בשבוע (מחזורי)",
        "dow_cos": "יום בשבוע (מחזורי)",
        "history_point_index": "נקודת זמן בהיסטוריה",
    }
    if feature in labels:
        return labels[feature]
    if feature.startswith("source__"):
        parts = feature.split("__")
        if len(parts) >= 3:
            metric = "נפח" if parts[-1] == "count" else "סנטימנט"
            source = parts[1].replace("_", " ").title()
            return f"{source} - {metric}"
    return feature.replace("_", " ")


def chart_importance(items: List[Dict], title: str, source_labels: Optional[Dict[str, str]] = None) -> Optional[go.Figure]:
    if not items:
        return None
    rows: List[Dict[str, Any]] = []
    for item in items[:12]:
        name = item.get("source") or item.get("feature") or ""
        if item.get("source") and source_labels:
            name = source_labels.get(str(name), str(name).replace("_", " ").title())
        elif item.get("feature"):
            name = _display_feature_name(str(name))
        rows.append({"name": name, "importance": float(item.get("importance") or 0.0)})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("importance")
    fig = px.bar(df, x="importance", y="name", orientation="h", color="importance",
                 color_continuous_scale="Teal")
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="חשיבות יחסית במודל",
        yaxis_title="",
        height=380,
        margin=dict(l=260, r=10, t=50, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="#fbfcfe",
    )
    fig.update_yaxes(automargin=True)
    return fig


def ml_predictions_frame(target_result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in target_result.get("predictions") or []:
        rows.append(
            {
                "שוק": row.get("market_question", ""),
                "הסתברות נוכחית (%)": (row.get("current_probability") or 0.0) * 100,
                "שינוי חזוי": row.get("predicted_delta"),
                "הסתברות חזויה (%)": (row.get("predicted_probability") or 0.0) * 100,
                "כיוון": row.get("predicted_direction"),
                "נפח כתבות": row.get("article_volume"),
                "סנטימנט ממוצע": row.get("mean_sentiment"),
            }
        )
    return pd.DataFrame(rows)


def _source_slug(source: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", source.strip().lower()).strip("_")
    return value or "unknown"


def source_sentiment_frame(articles: List[Dict]) -> pd.DataFrame:
    rows = [
        {
            "source": a.get("source") or "Unknown",
            "sentiment": float(a.get("sentiment") or 0.0),
        }
        for a in articles
        if a.get("sentiment") is not None
    ]
    if not rows:
        return pd.DataFrame(columns=["source", "article_count", "avg_sentiment"])
    df = pd.DataFrame(rows)
    return (
        df.groupby("source")
        .agg(article_count=("sentiment", "size"), avg_sentiment=("sentiment", "mean"))
        .reset_index()
        .sort_values("article_count", ascending=False)
    )


def chart_source_sentiment(source_df: pd.DataFrame) -> Optional[go.Figure]:
    if source_df.empty:
        return None
    df = source_df.sort_values("avg_sentiment")
    fig = px.bar(
        df,
        x="avg_sentiment",
        y="source",
        orientation="h",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
    )
    fig.update_layout(
        title=dict(text="סנטימנט ממוצע לפי כלי תקשורת", font=dict(size=15)),
        xaxis_title="מדד מתיחות ממוצע",
        yaxis_title="",
        coloraxis_showscale=False,
        height=380,
        margin=dict(l=220, r=10, t=50, b=10),
        plot_bgcolor="#fbfcfe",
    )
    fig.update_xaxes(range=[-1, 1], zeroline=True, zerolinecolor="#d0d4da")
    fig.update_yaxes(automargin=True)
    return fig


def chart_source_counts(source_df: pd.DataFrame) -> Optional[go.Figure]:
    if source_df.empty:
        return None
    df = source_df.sort_values("article_count")
    fig = px.bar(
        df,
        x="article_count",
        y="source",
        orientation="h",
        color="article_count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        title=dict(text="מספר כתבות לפי כלי תקשורת", font=dict(size=15)),
        xaxis_title="מספר כתבות",
        yaxis_title="",
        coloraxis_showscale=False,
        height=380,
        margin=dict(l=220, r=10, t=50, b=10),
        plot_bgcolor="#fbfcfe",
    )
    fig.update_yaxes(automargin=True)
    return fig


def _target_source_importance(ml_forecast: Dict[str, Any], horizon: str) -> Dict[str, float]:
    target = ((ml_forecast.get("targets") or {}).get(horizon) or {})
    return {
        str(item.get("source")): float(item.get("importance") or 0.0)
        for item in target.get("source_importance") or []
        if item.get("source")
    }


def source_reliability_frame(articles: List[Dict], ml_forecast: Dict[str, Any]) -> pd.DataFrame:
    source_df = source_sentiment_frame(articles)
    by_source = {
        str(row["source"]): {
            "article_count": int(row["article_count"]),
            "avg_sentiment": float(row["avg_sentiment"]),
        }
        for _, row in source_df.iterrows()
    }
    source_labels = ml_forecast.get("source_labels") or {}
    imp_1h = _target_source_importance(ml_forecast, "1h")
    imp_1d = _target_source_importance(ml_forecast, "1d")

    all_sources = set(by_source)
    all_sources.update(source_labels.values())
    rows: List[Dict[str, Any]] = []
    for source in sorted(all_sources):
        slug = _source_slug(source)
        one_hour = float(imp_1h.get(slug, 0.0))
        one_day = float(imp_1d.get(slug, 0.0))
        profile = by_source.get(source, {"article_count": 0, "avg_sentiment": 0.0})
        rows.append(
            {
                "כלי תקשורת": source,
                "מספר כתבות": profile["article_count"],
                "סנטימנט ממוצע": profile["avg_sentiment"],
                "חשיבות שעה": one_hour,
                "חשיבות יום": one_day,
                "ציון שימושיות חזויה": (one_hour + one_day) / 2,
            }
        )
    return pd.DataFrame(rows).sort_values("ציון שימושיות חזויה", ascending=False)


def _safe_corr(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[float]:
    sub = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3 or sub[x_col].nunique() <= 1 or sub[y_col].nunique() <= 1:
        return None
    return float(sub[x_col].corr(sub[y_col]))


def source_relationship_frame(ml_dataset: List[Dict], ml_forecast: Dict[str, Any]) -> pd.DataFrame:
    if not ml_dataset:
        return pd.DataFrame()
    df = pd.DataFrame(ml_dataset)
    source_labels = ml_forecast.get("source_labels") or {}
    rows: List[Dict[str, Any]] = []
    for slug, source in sorted(source_labels.items(), key=lambda item: item[1]):
        count_col = f"source__{slug}__count"
        sent_col = f"source__{slug}__sentiment"
        if count_col not in df.columns and sent_col not in df.columns:
            continue
        active = df
        if count_col in df.columns:
            active = df.loc[pd.to_numeric(df[count_col], errors="coerce").fillna(0) > 0]
        row: Dict[str, Any] = {
            "כלי תקשורת": source,
            "תצפיות עם כתבות": int(len(active)),
        }
        for horizon, label in (("1h", "שעה"), ("1d", "יום")):
            target_col = f"target_delta_{horizon}"
            row[f"מתאם נפח-{label}"] = (
                _safe_corr(active, count_col, target_col)
                if count_col in active.columns and target_col in active.columns
                else None
            )
            row[f"מתאם סנטימנט-{label}"] = (
                _safe_corr(active, sent_col, target_col)
                if sent_col in active.columns and target_col in active.columns
                else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def future_predictions_frame(ml_forecast: Dict[str, Any]) -> pd.DataFrame:
    targets = ml_forecast.get("targets") or {}
    merged: Dict[str, Dict[str, Any]] = {}
    for horizon in ("1h", "1d"):
        target = targets.get(horizon) or {}
        for prediction in target.get("predictions") or []:
            market_id = str(prediction.get("market_id") or prediction.get("market_question") or "")
            row = merged.setdefault(
                market_id,
                {
                    "שוק": prediction.get("market_question") or "",
                    "הסתברות נוכחית (%)": (prediction.get("current_probability") or 0.0) * 100,
                },
            )
            suffix = "שעה" if horizon == "1h" else "יום"
            row[f"שינוי חזוי בעוד {suffix}"] = prediction.get("predicted_delta")
            row[f"הסתברות חזויה בעוד {suffix} (%)"] = (
                (prediction.get("predicted_probability") or 0.0) * 100
            )
            row[f"כיוון בעוד {suffix}"] = {
                "up": "עלייה",
                "down": "ירידה",
                "flat": "יציב",
                None: "לא זמין",
            }.get(prediction.get("predicted_direction"), prediction.get("predicted_direction"))
    if not merged:
        return pd.DataFrame()
    return pd.DataFrame(merged.values())


def best_model_mae(ml_forecast: Dict[str, Any]) -> Optional[float]:
    result = primary_model_result(ml_forecast)
    return float(result["mae"]) if result.get("mae") is not None else None


def primary_model_result(ml_forecast: Dict[str, Any]) -> Dict[str, Any]:
    targets = ml_forecast.get("targets") or {}
    for horizon in ("1d", "1h"):
        result = targets.get(horizon) or {}
        if result.get("status") == "trained":
            return dict(result, horizon_label=horizon)
    return {}


def reliability_he(value: Optional[str]) -> str:
    return {
        "good": "אמינות גבוהה",
        "medium": "אמינות בינונית",
        "weak": "אמינות נמוכה",
    }.get(value or "", "לא זמין")


def direction_accuracy_text(value: Optional[float]) -> str:
    return f"{value:.1%}" if value is not None else "לא זמין"


def _pct_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) * 100
    except (TypeError, ValueError):
        return None


def _direction_distribution(future_df: pd.DataFrame) -> str:
    if future_df.empty or "כיוון בעוד יום" not in future_df.columns:
        return "אין מספיק תחזיות"
    counts = future_df["כיוון בעוד יום"].fillna("לא זמין").value_counts()
    if counts.empty:
        return "אין מספיק תחזיות"
    direction = str(counts.index[0])
    share = counts.iloc[0] / max(int(counts.sum()), 1)
    if direction == "יציב":
        return "רוב השווקים יציבים"
    return f"רוב השווקים בכיוון {direction} ({share:.0%})"


def forecast_status_line(model_result: Dict[str, Any], future_df: pd.DataFrame) -> str:
    reliability = reliability_he(model_result.get("reliability"))
    if not model_result:
        signal = "אין עדיין מודל מאומן"
    elif not model_result.get("beats_baseline"):
        signal = "אות חיזוי חלש"
    elif model_result.get("reliability") == "good":
        signal = "אות חיזוי חזק יחסית"
    else:
        signal = "אות חיזוי בינוני"
    return f"{signal}, {_direction_distribution(future_df)}, {reliability}."


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_static_table(df: pd.DataFrame, formats: Dict[str, str]) -> None:
    """Render a small HTML table instead of the canvas-based dataframe grid."""
    if df.empty:
        return
    shown = df.copy()
    for col, fmt in formats.items():
        if col in shown.columns:
            shown[col] = pd.to_numeric(shown[col], errors="coerce").map(
                lambda value: fmt.format(value) if pd.notna(value) else "—"
            )
    st.table(shown.reset_index(drop=True))


def top_source_name(reliability_df: pd.DataFrame) -> str:
    if reliability_df.empty:
        return "לא זמין"
    contributing = reliability_df.loc[
        pd.to_numeric(reliability_df["ציון שימושיות חזויה"], errors="coerce").fillna(0) > 0
    ]
    if contributing.empty:
        return "לא נמצאה תרומה מדידה"
    top = contributing.sort_values("ציון שימושיות חזויה", ascending=False).iloc[0]
    return str(top["כלי תקשורת"])


def compact_source_table(reliability_df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if reliability_df.empty:
        return pd.DataFrame()
    df = reliability_df.loc[
        pd.to_numeric(reliability_df["ציון שימושיות חזויה"], errors="coerce").fillna(0) > 0
    ]
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("ציון שימושיות חזויה", ascending=False).head(limit)
    return df.rename(
        columns={
            "כלי תקשורת": "Source",
            "מספר כתבות": "Number of Articles",
            "סנטימנט ממוצע": "Average Sentiment",
            "ציון שימושיות חזויה": "Predictive Score",
        }
    )[["Source", "Number of Articles", "Average Sentiment", "Predictive Score"]]


def chart_predictive_sources(reliability_df: pd.DataFrame, limit: int = 5) -> Optional[go.Figure]:
    table = compact_source_table(reliability_df, limit=limit)
    if table.empty:
        return None
    df = table.sort_values("Predictive Score")
    fig = px.bar(
        df,
        x="Predictive Score",
        y="Source",
        orientation="h",
        color="Predictive Score",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        title=dict(text="דירוג מקורות לפי תרומה לחיזוי", font=dict(size=15)),
        xaxis_title="Predictive Score",
        yaxis_title="",
        height=300,
        margin=dict(l=220, r=10, t=45, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="#fbfcfe",
    )
    fig.update_yaxes(automargin=True)
    return fig


def top_forecast_table(future_df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if future_df.empty:
        return pd.DataFrame()
    df = future_df.copy()
    if "שינוי חזוי בעוד יום" in df.columns:
        df["_rank_delta"] = pd.to_numeric(df["שינוי חזוי בעוד יום"], errors="coerce").abs()
    elif "שינוי חזוי בעוד שעה" in df.columns:
        df["_rank_delta"] = pd.to_numeric(df["שינוי חזוי בעוד שעה"], errors="coerce").abs()
    else:
        df["_rank_delta"] = 0
    df = df.sort_values("_rank_delta", ascending=False).head(limit)
    return df.rename(
        columns={
            "שוק": "Market",
            "הסתברות נוכחית (%)": "Current Probability",
            "הסתברות חזויה בעוד יום (%)": "Predicted Probability (24h)",
            "כיוון בעוד יום": "Direction",
        }
    )[["Market", "Current Probability", "Predicted Probability (24h)", "Direction"]]


def top_forecast_table_he(
    future_df: pd.DataFrame,
    model_result: Dict[str, Any],
    limit: int = 5,
) -> pd.DataFrame:
    table = top_forecast_table(future_df, limit=limit)
    if table.empty:
        return table
    table = table.rename(
        columns={
            "Market": "שוק",
            "Current Probability": "הסתברות נוכחית",
            "Predicted Probability (24h)": "הסתברות חזויה 24 שעות",
            "Direction": "כיוון",
        }
    )
    table["אות / אמינות"] = reliability_he(model_result.get("reliability"))
    return table[["שוק", "הסתברות נוכחית", "הסתברות חזויה 24 שעות", "כיוון", "אות / אמינות"]]


def main_conclusion_text(
    reliability_df: pd.DataFrame,
    future_df: pd.DataFrame,
    r: Optional[float],
    p: Optional[float],
    model_result: Dict[str, Any],
) -> str:
    contributing = (
        reliability_df.loc[
            pd.to_numeric(reliability_df["ציון שימושיות חזויה"], errors="coerce").fillna(0) > 0
        ]
        if not reliability_df.empty and "ציון שימושיות חזויה" in reliability_df.columns
        else pd.DataFrame()
    )
    if not contributing.empty:
        top_sources = (
            contributing.sort_values("ציון שימושיות חזויה", ascending=False)["כלי תקשורת"]
            .head(3)
            .tolist()
        )
        sources_sentence = (
            f"המקורות שתרמו לחיזוי בהרצה זו: {', '.join(top_sources)}. "
            "לשאר המקורות לא נמצאה תרומה חזויה מדידה."
        )
    else:
        sources_sentence = "בהרצה הנוכחית לא נמצאה תרומה חזויה מדידה לאף מקור תקשורת."

    trend_sentence = "אין עדיין מספיק תחזיות כדי לסכם את כיוון השווקים."
    if not future_df.empty and "כיוון בעוד יום" in future_df.columns:
        directions = future_df["כיוון בעוד יום"].value_counts()
        if not directions.empty:
            top_direction = str(directions.index[0])
            direction_text = {
                "עלייה": "עלייה קלה בהסתברות",
                "ירידה": "ירידה קלה בהסתברות",
                "יציב": "יציבות יחסית בהסתברות",
            }.get(top_direction, top_direction)
            trend_sentence = f"לפי הנתונים הנוכחיים, רוב השווקים מציגים צפי ל{direction_text} ב-24 השעות הקרובות."

    if r is None or p is None:
        pearson_sentence = "אין עדיין מספיק נתונים כדי להעריך מובהקות סטטיסטית של מתאם פירסון."
    elif p < 0.05:
        pearson_sentence = f"נמצא מתאם פירסון מובהק סטטיסטית (r={r:+.2f}, p={p:.3f})."
    else:
        pearson_sentence = f"נמצאה אינדיקציה לקשר של r={r:+.2f}, אך במדגם הנוכחי הקשר אינו מובהק סטטיסטית (p={p:.3f})."

    if model_result.get("beats_baseline"):
        ml_sentence = f"המודל מצליח לשפר את תחזית הבסיס בכ-{model_result.get('improvement_pct', 0):.1f}%."
    elif model_result:
        ml_sentence = "המודל עדיין אינו משפר משמעותית את תחזית הבסיס."
    else:
        ml_sentence = "אין עדיין תוצאת מודל מאומנת להשוואה מול תחזית בסיס."

    r2 = model_result.get("r2")
    if r2 is not None and r2 <= 0.02:
        ml_sentence += " ערך R² קרוב לאפס ולכן כוח ההסבר של המודל עדיין חלש."
    elif r2 is not None and r2 > 0.05:
        ml_sentence += " ערך R² חיובי מצביע על כוח הסבר מסוים."

    directional_accuracy = model_result.get("directional_accuracy")
    if directional_accuracy is not None and directional_accuracy > 0.60:
        ml_sentence += f" הדיוק הכיווני הוא {directional_accuracy:.1%}, ולכן קיימת אינדיקציה לאות כיווני שימושי."
    elif directional_accuracy is not None:
        ml_sentence += f" הדיוק הכיווני הוא {directional_accuracy:.1%}, ולכן חיזוי הכיוון עדיין מוגבל."

    return " ".join(part for part in (sources_sentence, trend_sentence, pearson_sentence, ml_sentence) if part)


def automatic_summary_frame(
    n_articles: int,
    n_markets: int,
    r: Optional[float],
    p: Optional[float],
    reliability_df: pd.DataFrame,
    model_result: Dict[str, Any],
) -> pd.DataFrame:
    significant = "כן" if p is not None and p < 0.05 else ("לא" if p is not None else "לא זמין")
    contributing = (
        reliability_df.loc[
            pd.to_numeric(reliability_df["ציון שימושיות חזויה"], errors="coerce").fillna(0) > 0
        ]
        if not reliability_df.empty and "ציון שימושיות חזויה" in reliability_df.columns
        else pd.DataFrame()
    )
    best_sources = (
        ", ".join(
            contributing.sort_values("ציון שימושיות חזויה", ascending=False)["כלי תקשורת"]
            .head(3)
            .tolist()
        )
        if not contributing.empty
        else "לא זמין"
    )
    return pd.DataFrame(
        [
            {"מדד": "מספר כתבות", "ערך": n_articles},
            {"מדד": "מספר שווקים", "ערך": n_markets},
            {"מדד": "Pearson r", "ערך": f"{r:+.3f}" if r is not None else "לא זמין"},
            {"מדד": "p-value", "ערך": f"{p:.4f}" if p is not None else "לא זמין"},
            {"מדד": "מובהקות סטטיסטית", "ערך": significant},
            {"מדד": "מקורות מובילים", "ערך": best_sources},
            {"מדד": "Random Forest MAE", "ערך": f"{model_result.get('mae'):.4f}" if model_result.get("mae") is not None else "לא זמין"},
            {"מדד": "Baseline MAE", "ערך": f"{model_result.get('baseline_mae'):.4f}" if model_result.get("baseline_mae") is not None else "לא זמין"},
            {"מדד": "שיפור מול baseline", "ערך": f"{model_result.get('improvement_pct'):+.1f}%" if model_result.get("improvement_pct") is not None else "לא זמין"},
            {"מדד": "R²", "ערך": f"{model_result.get('r2'):.4f}" if model_result.get("r2") is not None else "לא זמין"},
            {"מדד": "דיוק כיווני", "ערך": direction_accuracy_text(model_result.get("directional_accuracy"))},
            {"מדד": "תווית אמינות", "ערך": reliability_he(model_result.get("reliability"))},
        ]
    )


def validation_summary_frame(ml_forecast: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for horizon, label in (("1h", "שעה קדימה"), ("1d", "יום קדימה")):
        result = ((ml_forecast.get("targets") or {}).get(horizon) or {})
        report = result.get("validation_report") or {}
        rows.append(
            {
                "טווח חיזוי": label,
                "Train rows": report.get("train_rows", result.get("n_train")),
                "Test rows": report.get("test_rows", result.get("n_test")),
                "RF MAE": report.get("random_forest_mae", result.get("mae")),
                "Best baseline": report.get("best_baseline_name", result.get("best_baseline")),
                "Baseline MAE": report.get("best_baseline_mae", result.get("baseline_mae")),
                "Improvement": report.get("improvement_pct", result.get("improvement_pct")),
                "R²": report.get("r2", result.get("r2")),
                "Directional accuracy": _pct_value(report.get("directional_accuracy", result.get("directional_accuracy"))),
                "אמינות": reliability_he(report.get("reliability", result.get("reliability"))),
                "מסקנת תיקוף": report.get("conclusion_he", result.get("validation_conclusion_he", "")),
            }
        )
    return pd.DataFrame(rows)


def compact_validation_frame(model_result: Dict[str, Any]) -> pd.DataFrame:
    if not model_result:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Baseline MAE": model_result.get("baseline_mae"),
                "Random Forest MAE": model_result.get("mae"),
                "Improvement": model_result.get("improvement_pct"),
                "R²": model_result.get("r2"),
                "Directional accuracy": _pct_value(model_result.get("directional_accuracy")),
                "Reliability": reliability_he(model_result.get("reliability")),
            }
        ]
    )


def validation_interpretation(model_result: Dict[str, Any], p: Optional[float]) -> str:
    parts: List[str] = []
    if not model_result:
        parts.append("אין עדיין מודל מאומן להצגת תיקוף.")
    elif not model_result.get("beats_baseline"):
        parts.append("המודל אינו משפר את תחזית הבסיס במדד MAE, ולכן האינדיקציה החיזויית מוגבלת.")
    else:
        parts.append("המודל משפר את תחזית הבסיס במדד MAE, אך עדיין נדרשת בדיקה לאורך זמן.")

    r2 = model_result.get("r2")
    if r2 is not None and r2 <= 0.02:
        parts.append("כוח ההסבר לפי R² נמוך.")
    if p is not None and p >= 0.05:
        parts.append("מתאם Pearson אינו מובהק סטטיסטית במדגם הנוכחי.")
    return " ".join(parts)


def agent_markets_frame(agent_report: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in (agent_report.get("top_markets") or [])[:5]:
        rows.append(
            {
                "שוק": item.get("market"),
                "סטטוס": item.get("status"),
                "כיוון": item.get("direction"),
                "שינוי חזוי": item.get("predicted_delta"),
                "למה נבחר": item.get("why_selected"),
            }
        )
    return pd.DataFrame(rows)


def agent_sources_frame(agent_report: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in (agent_report.get("top_sources") or [])[:5]:
        rows.append(
            {
                "מקור": item.get("source"),
                "מספר כתבות": item.get("article_count"),
                "סנטימנט ממוצע": item.get("average_sentiment"),
                "ציון חזוי": item.get("predictive_score"),
                "הערה": item.get("note"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# UI building blocks
# --------------------------------------------------------------------------- #
def callout(headline: str, body_html: str, tone: str) -> None:
    bg, border = {
        "good": ("#eafaf0", "#bfe8cd"),
        "bad": ("#fdeceb", "#f6c9c5"),
        "neutral": ("#f3f4f6", "#e2e5ea"),
    }[tone]
    st.markdown(
        f"""
        <div class="callout" style="background:{bg};border-color:{border};">
          <h3>{headline}</h3><p>{body_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def howto(text_html: str) -> None:
    st.markdown(f"<div class='howto'>{text_html}</div>", unsafe_allow_html=True)


def sidebar() -> Dict[str, Any]:
    st.sidebar.title("בקרות")
    source = st.sidebar.radio(
        "מקור הנתונים", options=["Auto", "Firestore", "Local snapshot"], index=0,
        format_func=lambda x: DATA_SOURCE_OPTIONS_HE.get(x, x),
        help="מצב אוטומטי מעדיף את Firestore וחוזר לתמונת המצב המקומית בעת הצורך.",
    )
    snapshot_dir = st.sidebar.text_input("תיקיית תמונת המצב", value=DEFAULT_SNAPSHOT_DIR)
    keywords = st.sidebar.multiselect(
        "סינון לפי נושא", options=SETTINGS.geo_keywords, default=[],
        help="הצגת פריטים התואמים לפחות מילת מפתח אחת שנבחרה. ריק = הצגת הכול.",
    )
    if st.sidebar.button("טען מחדש נתונים"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("לרענון תמונת המצב הריצו:\n\n`python -m pipeline.run_pipeline --dry-run`")
    return {"source": source, "snapshot_dir": snapshot_dir, "keywords": keywords}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()

    st.title("האם החדשות תואמות את השווקים?")
    st.markdown(
        "לוח מחוונים זה משווה בין **מדד המתיחות הביטחונית בסיקור הגאופוליטי** (המחושב "
        "באמצעות מודל NLP לעיבוד שפה טבעית) לבין **ההסתברויות בשוקי הניבוי של Polymarket** "
        "עבור אירועים במזרח התיכון. כל שוק ניבוי מהווה נקודת נתונים אחת: אנו מצליבים בין "
        "*מידת ההסלמה או הרגיעה בסיקור הנושא* לבין *מה שהסוחרים מעריכים שיקרה*."
    )

    controls = sidebar()
    articles, markets, run, ml_dataset, agent_report, source_used = load_data(controls["source"], controls["snapshot_dir"])
    articles, markets = apply_keyword_filter(articles, markets, controls["keywords"])
    # Hard guards (independent of the data source): never display past-deadline
    # markets, and never display sports/entertainment markets (e.g. FIFA).
    markets = drop_past_markets(markets)
    markets = drop_excluded_topics(markets)

    last_updated = latest_data_timestamp(articles, markets, run)
    last_updated_he = (
        last_updated.strftime("%Y-%m-%d %H:%M UTC") if last_updated else "לא זמין"
    )
    st.caption(
        f"מקור הנתונים: **{SOURCE_LABELS_HE.get(source_used, source_used)}**"
        f" &nbsp;•&nbsp; עדכון נתונים אחרון: **{last_updated_he}**"
    )

    if not articles and not markets:
        st.info(
            "אין עדיין נתונים. ראשית יש להפיק תמונת מצב:\n\n"
            "```bash\npython -m pipeline.run_pipeline --dry-run\n```"
        )
        return

    if not markets:
        st.warning(
            "אין שווקים התואמים לסינון הנוכחי. נסו להסיר את הסינון לפי נושא "
            "או לבחור נושא אחר."
        )
        return

    # --- analysis ---------------------------------------------------------- #
    cross = cross_sectional_frame(articles, markets)
    r, p = cross_sectional_pearson(cross)
    avg_sent = average_sentiment(articles)
    avg_prob = average_probability(markets)
    n_markets_used = int(cross.dropna(subset=["sentiment", "probability"]).shape[0])
    ml_forecast = run.get("ml_forecast") or {}
    if not agent_report and ml_forecast:
        agent_report = generate_agent_report(articles, markets, run, ml_forecast)
    source_df = source_sentiment_frame(articles)
    reliability_df = source_reliability_frame(articles, ml_forecast) if ml_forecast else pd.DataFrame()
    future_df = future_predictions_frame(ml_forecast) if ml_forecast else pd.DataFrame()
    compact_sources = compact_source_table(reliability_df)
    model_result = primary_model_result(ml_forecast)
    mae = best_model_mae(ml_forecast)

    # --- Section 1: Executive summary ------------------------------------- #
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("כתבות", str(len(articles)))
    with k2:
        render_kpi_card("שווקים", str(len(markets)))
    with k3:
        render_kpi_card("המקור המשפיע ביותר", top_source_name(reliability_df))
    with k4:
        render_kpi_card("דיוק המודל (MAE)", f"{mae:.4f}" if mae is not None else "לא זמין")
    st.markdown(
        f"<div class='status-line'>{forecast_status_line(model_result, future_df)}</div>",
        unsafe_allow_html=True,
    )

    # --- Section 2: Automatic analytical summary ---------------------------- #
    st.divider()
    st.subheader("סיכום אנליטי אוטומטי")
    if agent_report:
        callout(
            "סיכום מצב החיזוי",
            (
                f"{agent_report.get('status', 'אין סטטוס זמין')} "
                f"מסקנה מעשית: {agent_report.get('suggested_next_action', 'מומלץ להמשיך לאסוף נתונים ולהריץ תיקוף נוסף.')}"
            ),
            "neutral" if agent_report.get("beats_baseline") else "bad",
        )
        st.caption(
            "שכבת הסיכום מנתחת את תוצאות המודל, מדדי התיקוף והתחזיות. היא אינה קובעת "
            "סיבתיות ואינה מנבאת את המציאות עצמה, אלא מפרשת תנועת הסתברות צפויה בפולימרקט."
        )
        summary_col1, summary_col2 = st.columns([1, 1.2])
        with summary_col1:
            src_agent_df = agent_sources_frame(agent_report)
            if not src_agent_df.empty:
                st.markdown("**מקורות בעלי תרומה גבוהה לחיזוי**")
                render_static_table(
                    src_agent_df.drop(columns=["הערה"], errors="ignore").head(3),
                    {"סנטימנט ממוצע": "{:+.3f}", "ציון חזוי": "{:.4f}"},
                )
            else:
                st.info("לא נמצאו מספיק נתונים להצגת רשימת מקורות מהימנה בהרצה הנוכחית.")
        with summary_col2:
            agent_market_df = agent_markets_frame(agent_report)
            if not agent_market_df.empty:
                st.markdown("**שווקים בולטים למעקב**")
                render_static_table(
                    agent_market_df.head(5),
                    {"שינוי חזוי": "{:+.3f}"},
                )
            else:
                st.info("לא נמצאו מספיק נתונים להצגת רשימת שווקים מהימנה בהרצה הנוכחית.")
        risks = agent_report.get("risks_and_limitations") or []
        if risks:
            st.caption("מגבלות מרכזיות: " + " ".join(risks[:2]))
    else:
        st.info("אין עדיין סיכום אנליטי אוטומטי. הריצו `python -m pipeline.run_pipeline --dry-run` כדי ליצור אותו.")

    # --- Section 3: Validation summary -------------------------------------- #
    st.divider()
    st.subheader("סיכום תיקוף המודלים")
    st.caption(
        "התיקוף נעשה בחלוקת זמן: המודל מתאמן על תצפיות מוקדמות ונבדק על תצפיות מאוחרות, "
        "כדי לדמות חיזוי עתידי ולא ערבוב של עבר ועתיד."
    )
    compact_validation = compact_validation_frame(model_result)
    if not compact_validation.empty:
        st.dataframe(
            compact_validation,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Baseline MAE": st.column_config.NumberColumn("Baseline MAE", format="%.4f"),
                "Random Forest MAE": st.column_config.NumberColumn("Random Forest MAE", format="%.4f"),
                "Improvement": st.column_config.NumberColumn("Improvement", format="%+.1f%%"),
                "R²": st.column_config.NumberColumn("R²", format="%.3f"),
                "Directional accuracy": st.column_config.NumberColumn("Directional accuracy", format="%.1f%%"),
            },
        )
        st.caption(validation_interpretation(model_result, p))
    else:
        st.info("אין עדיין תוצאות תיקוף ML זמינות.")

    # --- Section 4: Compact media source analysis ------------------------- #
    st.divider()
    st.subheader("ניתוח לפי כלי תקשורת")
    st.caption(
        "הטבלה מדרגת מקורות לפי שימושיות חזויה: עד כמה מאפייני המקור עזרו למודל לחזות "
        "תנועות הסתברות בפולימרקט. זה אינו דירוג אמינות עיתונאית."
    )
    src_table_col, src_chart_col = st.columns([1.15, 1])
    with src_table_col:
        if not compact_sources.empty:
            render_static_table(
                compact_sources,
                {"Average Sentiment": "{:+.3f}", "Predictive Score": "{:.4f}"},
            )
        else:
            st.info("אין עדיין מספיק נתוני ML לדירוג מקורות.")
    with src_chart_col:
        fig = chart_predictive_sources(reliability_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    # --- Section 5: Future forecast --------------------------------------- #
    st.divider()
    st.subheader("שווקים בולטים למעקב")
    st.caption(
        "התחזית מתייחסת לתנועת הסתברות בפולימרקט, ולא לתוצאה בעולם האמיתי. "
        "מוצגים חמשת השווקים עם השינוי החזוי הבולט ביותר."
    )
    top_forecasts_he = top_forecast_table_he(future_df, model_result, limit=5)
    if not top_forecasts_he.empty:
        render_static_table(
            top_forecasts_he,
            {"הסתברות נוכחית": "{:.1f}%", "הסתברות חזויה 24 שעות": "{:.1f}%"},
        )
        with st.expander("כל התחזיות"):
            st.dataframe(
                future_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "הסתברות נוכחית (%)": st.column_config.NumberColumn("הסתברות נוכחית (%)", format="%.1f%%"),
                    "שינוי חזוי בעוד שעה": st.column_config.NumberColumn("שינוי חזוי בעוד שעה", format="%+.3f"),
                    "שינוי חזוי בעוד יום": st.column_config.NumberColumn("שינוי חזוי בעוד יום", format="%+.3f"),
                    "הסתברות חזויה בעוד שעה (%)": st.column_config.NumberColumn("הסתברות חזויה בעוד שעה (%)", format="%.1f%%"),
                    "הסתברות חזויה בעוד יום (%)": st.column_config.NumberColumn("הסתברות חזויה בעוד יום (%)", format="%.1f%%"),
                },
            )
    else:
        st.info("אין עדיין תחזיות זמינות להצגה. הריצו מחדש את ה-pipeline עם נתוני ML מלאים.")

    # --- Section 6: Advanced analysis ------------------------------------- #
    st.divider()
    with st.expander("ניתוח מתקדם"):
        st.subheader("סיכום תוצאות מלא")
        st.dataframe(
            automatic_summary_frame(len(articles), len(markets), r, p, reliability_df, model_result),
            use_container_width=True,
            hide_index=True,
        )

        validation_df = validation_summary_frame(ml_forecast) if ml_forecast else pd.DataFrame()
        if not validation_df.empty:
            st.subheader("טבלת תיקוף מלאה")
            st.dataframe(
                validation_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "RF MAE": st.column_config.NumberColumn("RF MAE", format="%.4f"),
                    "Baseline MAE": st.column_config.NumberColumn("Baseline MAE", format="%.4f"),
                    "Improvement": st.column_config.NumberColumn("Improvement", format="%+.1f%%"),
                    "R²": st.column_config.NumberColumn("R²", format="%.3f"),
                    "Directional accuracy": st.column_config.NumberColumn("Directional accuracy", format="%.1f%%"),
                },
            )

        st.subheader("מסקנה מרכזית")
        callout(
            "פרשנות כוללת",
            main_conclusion_text(reliability_df, future_df, r, p, model_result),
            "neutral",
        )

        st.subheader("מדד המתיחות והמתאם")
        left, right = st.columns([1, 1.05])
        with left:
            st.plotly_chart(chart_sentiment_gauge(avg_sent), use_container_width=True)
            st.caption(
                f"{mood_label(avg_sent)} על בסיס {len(articles)} כתבות. "
                "הסקאלה: −1 = הסלמה ולחימה, +1 = רגיעה ודיפלומטיה."
            )
        with right:
            headline, sentence, tone = describe_correlation(r, p, n_markets_used)
            callout(headline, sentence, tone)

        st.subheader("גרפים סטטיסטיים")
        col_a, col_b = st.columns([1.25, 1])
        with col_a:
            st.plotly_chart(chart_scatter(cross, r, p), use_container_width=True)
        with col_b:
            heat = chart_heatmap(cross)
            if heat is not None:
                st.plotly_chart(heat, use_container_width=True)
            else:
                st.info("אין עדיין מספיק שונות בין השווקים לצורך הצגת מפת חום.")

        st.subheader("הסתברויות נוכחיות בשווקים")
        bar = chart_market_bar(markets)
        if bar is not None:
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("אין הסתברויות שוק זמינות עבור הסינון הנוכחי.")

        st.subheader("פירוט לפי כלי תקשורת")
        src_left, src_right = st.columns(2)
        with src_left:
            fig = chart_source_sentiment(source_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        with src_right:
            fig = chart_source_counts(source_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        if ml_forecast:
            rel_df = source_relationship_frame(ml_dataset, ml_forecast)
            if not rel_df.empty:
                st.markdown("**קשר בין כל מקור לבין תנועת הסתברות עתידית**")
                st.dataframe(
                    rel_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "מתאם נפח-שעה": st.column_config.NumberColumn("מתאם נפח-שעה", format="%+.3f"),
                        "מתאם סנטימנט-שעה": st.column_config.NumberColumn("מתאם סנטימנט-שעה", format="%+.3f"),
                        "מתאם נפח-יום": st.column_config.NumberColumn("מתאם נפח-יום", format="%+.3f"),
                        "מתאם סנטימנט-יום": st.column_config.NumberColumn("מתאם סנטימנט-יום", format="%+.3f"),
                    },
                )

            if not reliability_df.empty:
                st.markdown("**דירוג שימושיות חזויה מלא לפי מקור**")
                st.dataframe(
                    reliability_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "סנטימנט ממוצע": st.column_config.NumberColumn("סנטימנט ממוצע", format="%+.3f"),
                        "חשיבות שעה": st.column_config.NumberColumn("חשיבות שעה", format="%.4f"),
                        "חשיבות יום": st.column_config.NumberColumn("חשיבות יום", format="%.4f"),
                        "ציון שימושיות חזויה": st.column_config.NumberColumn("ציון שימושיות חזויה", format="%.4f"),
                    },
                )

            st.subheader("אבחון מודל ML")
            m1, m2, m3 = st.columns(3)
            m1.metric("תצפיות בדאטהסט", ml_forecast.get("n_observations", 0))
            m2.metric("שווקים בדאטהסט", ml_forecast.get("n_markets", 0))
            m3.metric("חלון כתבות", f"{ml_forecast.get('article_window_hours', 24)} שעות")

            st.subheader("תיקוף מודלים מורחב")
            targets = ml_forecast.get("targets") or {}
            validation_tabs = st.tabs(["תיקוף שעה", "תיקוף יום"])
            for tab, horizon in zip(validation_tabs, ("1h", "1d")):
                with tab:
                    result = targets.get(horizon) or {}
                    report = result.get("validation_report") or {}
                    if not report:
                        st.info("אין דוח תיקוף מפורט עבור יעד זה.")
                        continue
                    callout(
                        "מסקנת תיקוף",
                        report.get("conclusion_he", "אין מסקנה זמינה."),
                        "neutral" if result.get("beats_baseline") else "bad",
                    )
                    st.caption(report.get("leakage_note_he", "הפיצ'רים נבנים רק ממידע שקדם לנקודת החיזוי."))

                    err = report.get("error_distribution") or {}
                    if err:
                        st.markdown("**התפלגות שגיאות**")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {"מדד": "Mean Error", "ערך": err.get("mean_error")},
                                    {"מדד": "Median Absolute Error", "ערך": err.get("median_absolute_error")},
                                    {"מדד": "P90 Absolute Error", "ערך": err.get("p90_absolute_error")},
                                    {"מדד": "Max Absolute Error", "ערך": err.get("max_absolute_error")},
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                            column_config={"ערך": st.column_config.NumberColumn("ערך", format="%.4f")},
                        )

                    moving_subset = report.get("moving_subset") or {}
                    if moving_subset:
                        st.markdown("**תיקוף על תצפיות שבהן השוק זז**")
                        moving_subset_df = pd.DataFrame(
                            [
                                {
                                    "סטטוס": moving_subset.get("status"),
                                    "תצפיות בדיקה": moving_subset.get("n_test"),
                                    "חלק מהמדגם": _pct_value(moving_subset.get("share_of_test")),
                                    "Random Forest MAE": moving_subset.get("random_forest_mae"),
                                    "Baseline MAE": moving_subset.get("baseline_mae"),
                                    "Improvement": moving_subset.get("improvement_pct"),
                                    "R²": moving_subset.get("r2"),
                                    "Directional accuracy": _pct_value(moving_subset.get("directional_accuracy")),
                                }
                            ]
                        )
                        st.dataframe(
                            moving_subset_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "חלק מהמדגם": st.column_config.NumberColumn("חלק מהמדגם", format="%.1f%%"),
                                "Random Forest MAE": st.column_config.NumberColumn("Random Forest MAE", format="%.4f"),
                                "Baseline MAE": st.column_config.NumberColumn("Baseline MAE", format="%.4f"),
                                "Improvement": st.column_config.NumberColumn("Improvement", format="%+.1f%%"),
                                "R²": st.column_config.NumberColumn("R²", format="%.3f"),
                                "Directional accuracy": st.column_config.NumberColumn("Directional accuracy", format="%.1f%%"),
                            },
                        )

                    topic_perf = report.get("topic_performance") or []
                    if topic_perf:
                        st.markdown("**ביצועים לפי נושא**")
                        topic_perf_df = pd.DataFrame(topic_perf)
                        if "directional_accuracy" in topic_perf_df.columns:
                            topic_perf_df["directional_accuracy"] = pd.to_numeric(
                                topic_perf_df["directional_accuracy"], errors="coerce"
                            ) * 100
                        st.dataframe(
                            topic_perf_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "mae": st.column_config.NumberColumn("MAE", format="%.4f"),
                                "directional_accuracy": st.column_config.NumberColumn("Directional Accuracy", format="%.1f%%"),
                            },
                        )

                    source_validation = report.get("source_validation") or []
                    if source_validation:
                        st.markdown("**תיקוף לפי מקור תקשורת**")
                        st.dataframe(
                            pd.DataFrame(source_validation),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "predictive_importance": st.column_config.NumberColumn("Predictive Importance", format="%.4f"),
                            },
                        )

            source_labels = ml_forecast.get("source_labels") or {}
            tab_1h, tab_1d = st.tabs(["חיזוי לשעה הקרובה", "חיזוי ליום הקרוב"])
            for tab, horizon, title in (
                (tab_1h, "1h", "שינוי הסתברות בשעה הקרובה"),
                (tab_1d, "1d", "שינוי הסתברות ביום הקרוב"),
            ):
                with tab:
                    result = targets.get(horizon) or {}
                    if result.get("status") != "trained":
                        st.info(
                            "אין עדיין מספיק תצפיות מאומנות עבור יעד זה. "
                            "הרצה עם היסטוריית מחירים וכתבות ארוכה יותר תשפר את הכיסוי."
                        )
                        st.caption(f"סטטוס מודל: {result.get('status', 'לא זמין')}")
                        continue

                    q1, q2, q3, q4 = st.columns(4)
                    q1.metric("MAE", f"{result.get('mae', 0):.4f}")
                    q2.metric("Baseline MAE", f"{result.get('baseline_mae'):.4f}" if result.get("baseline_mae") is not None else "לא זמין")
                    q3.metric("שיפור מול בסיס", f"{result.get('improvement_pct'):+.1f}%" if result.get("improvement_pct") is not None else "לא זמין")
                    q4.metric("אמינות", reliability_he(result.get("reliability")))
                    r1, r2_col, r3 = st.columns(3)
                    r1.metric("R²", f"{result.get('r2'):.3f}" if result.get("r2") is not None else "לא זמין")
                    r2_col.metric("דיוק כיווני", direction_accuracy_text(result.get("directional_accuracy")))
                    r3.metric("תצפיות בדיקה", result.get("n_test", 0))

                    baselines = result.get("baselines") or {}
                    if baselines:
                        baselines_df = pd.DataFrame(
                            [
                                {
                                    "Baseline": name,
                                    "MAE": values.get("mae"),
                                    "Directional Accuracy": _pct_value(values.get("directional_accuracy")),
                                }
                                for name, values in baselines.items()
                            ]
                        )
                        st.dataframe(
                            baselines_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "MAE": st.column_config.NumberColumn("MAE", format="%.4f"),
                                "Directional Accuracy": st.column_config.NumberColumn("Directional Accuracy", format="%.1f%%"),
                            },
                        )

                    imp_col, src_col = st.columns(2)
                    with imp_col:
                        fig = chart_importance(result.get("feature_importance") or [], f"מאפיינים חשובים - {title}")
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                    with src_col:
                        fig = chart_importance(
                            result.get("source_importance") or [],
                            f"תרומת מקורות תקשורת - {title}",
                            source_labels=source_labels,
                        )
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)

                    pdf = ml_predictions_frame(result)
                    if not pdf.empty:
                        st.dataframe(
                            pdf,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "הסתברות נוכחית (%)": st.column_config.NumberColumn("הסתברות נוכחית (%)", format="%.1f%%"),
                                "שינוי חזוי": st.column_config.NumberColumn("שינוי חזוי", format="%+.3f"),
                                "הסתברות חזויה (%)": st.column_config.NumberColumn("הסתברות חזויה (%)", format="%.1f%%"),
                                "סנטימנט ממוצע": st.column_config.NumberColumn("סנטימנט ממוצע", format="%+.2f"),
                            },
                        )
        else:
            st.info("תוצאות ML אינן זמינות עדיין.")

        st.subheader("עיון בנתוני הגלם")
        tab_articles, tab_markets = st.tabs(["כתבות", "שווקים"])

        with tab_articles:
            if articles:
                adf = pd.DataFrame(
                    [
                        {
                            "תאריך פרסום": (a.get("published_at") or "לא ידוע")[:16].replace("T", " "),
                            "מקור": a.get("source") or "לא ידוע",
                            "מדד מתיחות": a.get("sentiment", 0.0),
                            "סיווג טון": a.get("sentiment_label", ""),
                            "כותרת": a.get("title", ""),
                            "נושאים": ", ".join(a.get("matched_keywords", [])),
                            "קישור": a.get("url", ""),
                        }
                        for a in articles
                    ]
                ).sort_values("תאריך פרסום", ascending=False)
                st.dataframe(
                    adf, use_container_width=True, hide_index=True,
                    column_config={
                        "קישור": st.column_config.LinkColumn("קישור", display_text="פתח"),
                        "מדד מתיחות": st.column_config.NumberColumn("מדד מתיחות", format="%+.2f"),
                    },
                )
            else:
                st.info("אין כתבות עבור הסינון הנוכחי.")

        with tab_markets:
            if markets:
                mdf = pd.DataFrame(
                    [
                        {
                            "שאלה": m.get("question", ""),
                            "הסתברות (%)": (m.get("current_probability") or 0.0) * 100,
                            "נפח מסחר": m.get("volume"),
                            "נקודות היסטוריה": m.get("history_points", len(m.get("price_history", []) or [])),
                            "נושאים": ", ".join(m.get("matched_keywords", [])),
                            "קישור": m.get("url", ""),
                        }
                        for m in markets
                    ]
                ).sort_values("נפח מסחר", ascending=False, na_position="last")
                st.dataframe(
                    mdf, use_container_width=True, hide_index=True,
                    column_config={
                        "קישור": st.column_config.LinkColumn("קישור", display_text="פתח"),
                        "הסתברות (%)": st.column_config.NumberColumn("הסתברות (%)", format="%.1f%%"),
                        "נפח מסחר": st.column_config.NumberColumn("נפח מסחר", format="$%.0f"),
                    },
                )
            else:
                st.info("אין שווקים עבור הסינון הנוכחי.")

    finished = run.get("finished_at")
    if finished:
        st.caption(f"הרצת ה-pipeline האחרונה: {finished}  •  מזהה הרצה: {run.get('run_id', 'לא ידוע')}")

    # --- methodology & tech-stack footer ---------------------------------- #
    st.divider()
    st.markdown(
        """
        <div class="methodology">
          <h4>מתודולוגיה ומחסנית טכנולוגית</h4>
          <p>מערכת זו פותחה במסגרת פרויקט גמר בלמידת מכונה. הנתונים נשאבים בזמן אמת
          ממקורות חדשות ומ-Polymarket API, מאוחסנים ב-Firebase Firestore, ומנותחים
          באמצעות מודל NLP (Hugging Face) לניתוח סנטימנט.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
