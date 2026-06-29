# PROJECT_STATUS_FOR_CODEX.md

> Handoff document for an AI coding assistant (or new developer) picking up this
> project. Written after a full read of every source file on 2026-06-15.
> **Precise file, class, and function names are used throughout.**

---

## 1. Project goal

The code in this folder is a **research pipeline + dashboard** that asks one
question: *does the tone of news coverage move in step with prediction-market
prices?*

As **actually implemented**, it does the following for **Middle-East
geopolitics** (Israel / Gaza / Iran / Hezbollah / Hamas, etc.):

1. Scrape recent news articles from several RSS feeds.
2. Score each article's sentiment with a Hugging Face NLP model.
3. Pull matching **Polymarket** prediction markets (current implied probability
   + price history).
4. Compute a **Pearson correlation** between news sentiment and market
   probability (both a daily time-series version and a cross-sectional version).
5. Persist everything to **Firebase Firestore** and a local JSON snapshot.
6. Present the results in a **Hebrew, right-to-left Streamlit dashboard**.

> ⚠️ **Read Section 2 carefully.** The original course work-plan describes a
> *different* domain and methods than what is built. This is the single most
> important thing for the next contributor to understand.

---

## 2. Teacher's requirements — and a major scope discrepancy

### 2a. Original work-plan (from `../תוכנית עבודה לפרויקט מעשי בלמידת מכונה.pdf`)

The course assignment ("Project 4", authors: Edward Nepomniashchi, Einav Dar'i,
Agam Choulio) specifies:

- **Research question:** To what extent does **social-media activity (Twitter/X)
  of "whales"/influencers** affect Polymarket prices?
- **Domain:** **SPORTS markets** on Polymarket (games, championships, player
  transfers) — explicitly the volatile sports vertical.
- **Text sources:** **Twitter/X API (Basic)** and/or **Reddit API / RSS**
  (`r/sportsbook`, `r/nba`).
- **Sentiment model:** Hugging Face **`cardiffnlp/twitter-roberta-base-sentiment`**
  (a tweet-trained model with a true neutral class).
- **Statistics:** **Granger Causality** (directional), plus Pearson/Spearman
  correlation.
- **Data:** ≥ 3 months of history for **5–10 active sports markets**.
- **Deliverables / success:** statistically significant correlation (p < 0.05)
  in ≥ 3 sports markets; stretch: directional price prediction > 60% accuracy;
  well-documented **GitHub** repo runnable **in one command**, a README,
  technical design doc, and a landing page. Stretch extensions: **whale
  identification** and **influencer accuracy scoring**.

### 2b. What the code actually does (the pivot)

| Aspect | Work-plan (PDF) | Implemented in this repo |
|---|---|---|
| Domain | **Sports** markets | **Geopolitics** (Middle East) |
| Text source | Twitter/X, Reddit | **News RSS** (BBC, CNN, TOI, JPost, Fox, NYT) |
| Sentiment model | `cardiffnlp/twitter-roberta-base-sentiment` | **`distilbert-base-uncased-finetuned-sst-2-english`** (see `Settings.sentiment_model` in `config/settings.py`) |
| Statistics | **Granger causality** + correlation | **Pearson only** (`build_correlation_summary` in `pipeline/run_pipeline.py`; `cross_sectional_pearson` in `frontend/app.py`) |
| Whales / influencers | Required extension | **Not present** (no social data) |

Notably, recent dashboard work deliberately **excludes** sports markets
(`_EXCLUDED_TERMS` in `data/polymarket_client.py` rejects FIFA / World Cup /
football / sports / Olympics) — the *opposite* of the PDF's sports focus.

**Action for the next contributor:** confirm with the professor whether the
geopolitics/news pivot is approved. The rest of this document assumes the
geopolitics implementation is the thing being maintained, but flags everywhere
the gap from the written plan matters.

### 2c. "Academic polish" requirements added during final-presentation prep

These were requested most recently and **are implemented** in `frontend/app.py`
and `data/polymarket_client.py`:

- Dark-mode-safe KPI metric cards (removed custom white CSS on
  `div[data-testid="stMetric"]`).
- Strictly active, future-dated markets only.
- Robust article date parsing (no more "Unknown").
- "Last updated" timestamp (`latest_data_timestamp`) and a Hebrew
  methodology/tech-stack footer.
- Full RTL layout (`body, .stApp { direction: rtl; ... }`).
- Rebrand of "news sentiment/mood" → **"מדד המתיחות הביטחונית"** (security-tension
  index): −1 = "הסלמה ולחימה", +1 = "רגיעה ודיפלומטיה".
- Bar-chart Y-axis labels no longer truncated (`automargin=True`, `margin l=400`).
- Negative keyword + past-date market filtering at both data and app layers.

---

## 3. Important files and what each does

### Entry points
- **`pipeline/run_pipeline.py`** — end-to-end orchestrator. Key functions:
  `run(...)` (5 stages: news → sentiment → markets → correlate → persist),
  `build_correlation_summary` / `_daily_sentiment` / `_daily_probability`
  (daily Pearson), `write_local_snapshot`, and the `argparse` CLI in
  `_parse_args` / `main`. Run with `python -m pipeline.run_pipeline`.
- **`frontend/app.py`** — the Streamlit dashboard (the demo surface). Key
  functions: `main`, `inject_css` (RTL + theming), `load_data` /
  `load_local_snapshot` / `load_firestore`, `drop_past_markets` /
  `drop_excluded_topics` (app-side guards), `latest_data_timestamp`,
  `cross_sectional_frame` / `cross_sectional_pearson` (the dashboard's own
  cross-sectional analysis — one observation per market), and chart builders
  `chart_sentiment_gauge`, `chart_scatter`, `chart_heatmap`, `chart_market_bar`.
  Run with `streamlit run frontend/app.py`.

### Data layer (`data/`)
- **`data/news_scraper.py`** — class **`NewsScraper`**. Fetches + keyword-filters
  RSS articles. Important methods: `fetch_articles`, `_parse_feed` (now handles
  both RSS `<item>` and Atom `<entry>`), `_parse_date` / `_extract_date` /
  `_localname` / `_children_by_localname` / `_extract_link` (robust date + link
  extraction), `_matched_keywords`, `fetch_full_text`, `analysis_text`.
- **`data/polymarket_client.py`** — class **`PolymarketClient`**. Talks to the
  Gamma API (markets) and CLOB API (price history). Important methods:
  `collect`, `list_geopolitical_markets`, `_normalise_market`,
  `_is_active_market` + `_parse_iso` (drops closed/resolved/past-deadline),
  `_has_excluded_topic` + `_EXCLUDED_TERMS` (rejects sports/entertainment),
  `_matched_keywords`, `get_price_history`.
- **`data/firebase_client.py`** — class **`FirestoreClient`** (+
  `FirestoreClientError`). Idempotent `merge=True` upserts. Methods:
  `upsert_documents`, `save_articles`, `save_markets`, `save_run`,
  `get_articles`, `get_markets`, `healthcheck`. Collections come from
  `Settings`: `bbc_articles`, `polymarket_markets`, `polymarket_price_history`,
  `pipeline_runs`.

### ML layer (`ml/`)
- **`ml/sentiment_analyzer.py`** — class **`SentimentAnalyzer`**. Lazy-loads a
  Hugging Face `pipeline`. Methods: `analyze`, `analyze_batch`,
  `analyze_articles` (annotates each article with `sentiment_label`,
  `sentiment_score`, `sentiment`), and the static `_to_signed` that maps a model
  label+confidence to a signed score in [−1, 1]. **This is the core ML math —
  do not change it without reason (see Section 8).**

### Config (`config/`)
- **`config/settings.py`** — Pydantic **`Settings`** singleton via
  `get_settings()`. Holds `geo_keywords` (`DEFAULT_GEO_KEYWORDS`), `news_feeds`
  (`DEFAULT_NEWS_FEEDS`), `sentiment_model`, `lookback_days` (30),
  `max_markets` (25), `price_interval`/`price_fidelity`, Firestore collection
  names, and `has_firebase_credentials`. **Central — changes ripple everywhere.**
- **`config/logging_config.py`** — `configure_logging` (JSON or text) and
  `get_logger`.

### Data / artifacts
- **`output/articles.json`, `output/markets.json`, `output/run.json`** —
  generated snapshot consumed by the dashboard when Firestore is unavailable.
  **Generated; do not hand-edit.** Latest `run.json`: 146 articles, 25 markets,
  `pearson_r = -0.2977`, `p_value = 0.2459`.
- **`.env`** and **`firebase_creds.json`** — runtime secrets (see Section 8).
- **`requirements.txt`** — pinned deps (Python 3.11).

---

## 4. What is already implemented (working)

- **Full automated pipeline** (`pipeline/run_pipeline.py`): multi-source news
  ingest → HF sentiment scoring → Polymarket markets + CLOB price history →
  Pearson correlation → JSON snapshot → optional Firestore persistence, with a
  flexible CLI (`--dry-run`, `--lookback-days`, `--max-markets`, `--full-text`,
  `--skip-articles`, `--skip-markets`, `--output-dir`).
- **Working sentiment model** (`SentimentAnalyzer`) with batching and label
  normalization across SST-2 / `LABEL_x` / star-rating conventions.
- **Resilient HTTP** (retry/backoff sessions) in both API clients.
- **Robust RSS/Atom parsing** with reliable publication dates.
- **Strict market hygiene**: active + future-dated only, sports/entertainment
  excluded, at both the data layer and the app layer.
- **Polished Streamlit dashboard**: KPIs, security-tension gauge, scatter +
  trend line, correlation heatmap, market-probability bar chart, raw-data
  tables, RTL Hebrew UI, last-updated timestamp, methodology footer.
- **Firestore persistence layer** with idempotent upserts and a healthcheck.
- **Centralized config + structured logging.**
- Snapshot fallback means the dashboard runs **without** Firebase credentials.

---

## 5. What is missing

Relative to the **work-plan (Section 2a)**:

- **Granger causality** — required sub-goal, extension, and success metric. Only
  Pearson exists. No `statsmodels` dependency.
- **Social-media data** (Twitter/X, Reddit) — none. Only news RSS.
- **`cardiffnlp/twitter-roberta-base-sentiment`** — plan's model; repo uses
  DistilBERT SST-2 instead.
- **Whale / influencer identification** and **accuracy scoring** — not started.
- **≥ 3 months of history for 5–10 sports markets** — current run is a 30-day
  window of geopolitics markets; only a single snapshot exists.
- **Spearman correlation** — plan allows Pearson *or* Spearman; only Pearson.

Engineering/process gaps:

- **No README** and **no landing page** (both explicitly required).
- **No automated tests** (no `tests/`, no `pytest` config).
- **Not a git repository** here (no `.git`), despite the GitHub requirement.
- **Statistical power**: latest correlation is **not significant**
  (`p_value = 0.2459 > 0.05`), so the minimum success metric is not yet met.

---

## 6. Known bugs / risks

1. **Domain mismatch with the assignment (highest priority).** Sports + Twitter +
   Granger (plan) vs geopolitics + news + Pearson (code). Needs professor
   sign-off or a documented pivot.
2. **🔐 Secrets committed in the project folder.** `firebase_creds.json` contains
   a **real service-account private key** (`project_id: polymarket-7bfb7`), and
   `.env` holds live config. If this folder is pushed to GitHub as required, the
   key leaks. Rotate the key, add a `.gitignore`, and never commit these.
3. **Correlation not statistically significant** in the current data
   (`p = 0.2459`). The headline result may not support the research claim yet.
4. **Two different correlation methods** can confuse reviewers:
   `pipeline/run_pipeline.py` computes a **daily time-series** Pearson, while
   `frontend/app.py` (`cross_sectional_frame` / `cross_sectional_pearson`)
   computes a **cross-sectional** Pearson (one point per market). They can
   disagree; document which is the "official" result.
5. **DistilBERT SST-2 has no neutral class.** `_to_signed` therefore returns
   `±score` for almost all real text (never exactly 0 unless the input is
   empty), pushing war coverage toward sentiment extremes. The plan's
   `twitter-roberta` model has a genuine neutral class and is better suited.
6. **Feed fragility.** CNN's feed (`http://rss.cnn.com/rss/edition_meast.rss`)
   currently returns **HTTP 403**; it produced the earlier "Unknown" dates.
   Al Jazeera and Reuters were already dropped (see comments in
   `config/settings.py`). Source availability varies by network.
7. **Rebrand semantic subtlety.** The "מדד המתיחות" (tension index) scale is
   inverted vs intuition: **+1 = calm/diplomacy**, **−1 = escalation**. Be ready
   to explain that a *higher* index means *less* tension.
8. **Snapshot can lag code.** App-side guards (`drop_past_markets`,
   `drop_excluded_topics`) exist precisely because `output/*.json` may have been
   generated by older logic. Re-run the pipeline to refresh.
9. **No retry/limit on Firestore reads** (`_read_collection` streams the whole
   collection); fine at current scale, but unbounded as data grows.

---

## 7. How to build / run / test

**Prerequisites:** Python 3.11. A `venv/` already exists at the parent folder.

```bash
# from the Polymarket_Geopolitics/ directory
python -m venv venv && source venv/bin/activate      # if not using the existing venv
pip install -r requirements.txt
```

**Configure** (already present, but for a fresh clone): copy/set `.env` with
`GOOGLE_APPLICATION_CREDENTIALS` (path to `firebase_creds.json`) and
`FIREBASE_PROJECT_ID`. The dashboard also runs with **no** credentials by using
the local snapshot.

**Run the pipeline:**
```bash
python -m pipeline.run_pipeline --dry-run      # all stages, local snapshot only, no Firestore writes
python -m pipeline.run_pipeline                # full run, writes to Firestore
python -m pipeline.run_pipeline --lookback-days 14 --max-markets 10 --full-text
```
> First run downloads the Hugging Face model (~250 MB) — expect a slow cold start.

**Run the dashboard:**
```bash
streamlit run frontend/app.py
```

**"Testing" today (no test suite yet)** — each module has a smoke-test entry
point:
```bash
python -m data.news_scraper          # prints recent matched articles
python -m data.polymarket_client     # prints a few markets + price-point counts
python -m ml.sentiment_analyzer      # prints signed sentiment for 2 sample texts
python -m data.firebase_client       # prints "Firestore reachable: True/False"
python -m py_compile frontend/app.py data/polymarket_client.py data/news_scraper.py
```

---

## 8. Files that should NOT be changed unless necessary

- **`firebase_creds.json`** — live secret. Never edit, print, or commit. Rotate
  if exposed.
- **`.env`** — live secret/config. Keep out of version control.
- **`ml/sentiment_analyzer.py`** — the ML/NLP math (`_to_signed`, scoring). The
  project owner explicitly asked not to change the underlying ML/NLP logic.
  Changing the **model name** is fine via `Settings.sentiment_model` (config),
  but leave the scoring functions alone unless that is the task.
- **`config/settings.py`** — central, validated config consumed by every module.
  Safe to add fields; risky to rename/remove existing ones.
- **`output/*.json`** — generated artifacts. Regenerate via the pipeline; do not
  hand-edit (the dashboard parses them directly).
- The **statistical core** (`build_correlation_summary` and the `cross_sectional_*`
  functions) — keep stable so results stay reproducible; extend alongside rather
  than rewrite.

---

## 9. Recommended next steps (small, safe stages)

Ordered so each stage is independently shippable and low-risk.

**Stage 0 — Reconcile scope (docs only, no code).**
Get explicit professor sign-off on geopolitics-vs-sports. Either update the
work-plan to match the implementation, or open a plan to re-align. Nothing below
is worth much until this is settled.

**Stage 1 — Repo hygiene & secrets (no logic change).**
Add a `.gitignore` (ignore `.env`, `firebase_creds.json`, `venv/`, `__pycache__/`,
optionally `output/`). **Rotate the Firebase key** if the folder was ever shared.
`git init` and make a clean first commit *after* secrets are ignored.

**Stage 2 — Documentation (no logic change).**
Write a `README.md` (goal, architecture diagram, setup, run commands, screenshot)
and the required landing page. This directly satisfies a graded requirement and
costs no runtime risk.

**Stage 3 — Add Granger causality (additive).**
Add `statsmodels` to `requirements.txt`; add a `build_granger_summary(...)`
beside `build_correlation_summary` in `pipeline/run_pipeline.py` operating on the
existing daily frames (`_daily_sentiment`, `_daily_probability`). Surface the
result in `frontend/app.py` as a new callout. Keep Pearson untouched.

**Stage 4 — Add a test suite (additive).**
Create `tests/` with `pytest` covering pure logic: `NewsScraper._parse_date` /
`_parse_feed`, `PolymarketClient._is_active_market` / `_has_excluded_topic`,
`SentimentAnalyzer._to_signed`, and `build_correlation_summary`. These run
offline with fixtures — no network or model download.

**Stage 5 — Model alignment (config-first).**
Optionally set `SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment` to
match the plan and gain a neutral class. `_to_signed` already handles its labels,
so this is a config change; validate on a sample before committing.

**Stage 6 — Strengthen the result.**
Widen `lookback_days` toward the 3-month requirement and lean on the daily
time-series correlation (and Spearman) so the significance test has more power
than the current cross-sectional snapshot (`p = 0.2459`).

**Stage 7 (stretch) — Plan extensions.**
Only after the above: influencer/"whale" analysis, which presupposes a
social-media data source that does not yet exist.

---

*End of status document.*
