"""NLP sentiment analysis of news articles from every configured outlet.

Wraps a Hugging Face ``transformers`` text-classification pipeline.  The model
is loaded lazily on first use (model download / weight load is expensive), and
the tokenizer is reused for length-safe truncation.

Each analysed item is annotated with:

* ``sentiment_label``  — model label (e.g. ``POSITIVE`` / ``NEGATIVE``).
* ``sentiment_score``  — model confidence in ``[0, 1]``.
* ``sentiment``        — signed score in ``[-1, 1]`` (negative = bearish tone),
  the value correlated against market probabilities downstream.

Usage::

    from ml.sentiment_analyzer import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    scored = analyzer.analyze_articles(articles)   # adds sentiment_* fields
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Lazy-loaded Hugging Face sentiment pipeline with batch helpers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
        force_cpu: Optional[bool] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.sentiment_model
        self.force_cpu = self.settings.sentiment_force_cpu if force_cpu is None else force_cpu
        self._pipeline = None
        self._tokenizer = None
        self._max_length = 512

    # ------------------------------------------------------------------ #
    # Lazy initialisation
    # ------------------------------------------------------------------ #
    def _resolve_device(self) -> int:
        """Return the transformers device id: GPU 0 if available else -1 (CPU)."""
        if self.force_cpu:
            return -1
        try:
            import torch

            if torch.cuda.is_available():
                return 0
        except Exception:  # pragma: no cover - torch always present per requirements
            pass
        return -1

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers/torch are not installed. Run "
                "`pip install -r requirements.txt`."
            ) from exc

        device = self._resolve_device()
        logger.info(
            "Loading sentiment model '%s' on %s.",
            self.model_name,
            "GPU" if device >= 0 else "CPU",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._max_length = min(getattr(self._tokenizer, "model_max_length", 512) or 512, 512)
        self._pipeline = pipeline(
            task="sentiment-analysis",
            model=model,
            tokenizer=self._tokenizer,
            device=device,
            truncation=True,
            max_length=self._max_length,
        )

    # ------------------------------------------------------------------ #
    # Core scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_signed(label: str, score: float) -> float:
        """Map (label, confidence) to a signed sentiment in [-1, 1].

        Handles the common Hugging Face conventions:
        * SST-2 style: ``POSITIVE`` / ``NEGATIVE``.
        * Generic heads: ``LABEL_0`` (neg) / ``LABEL_1`` (pos).
        * Star ratings (nlptown): ``"1 star"`` .. ``"5 stars"`` → 1-2 neg,
          3 neutral, 4-5 positive.
        """
        label_upper = (label or "").upper().strip()

        if "STAR" in label_upper:
            stars = next((int(c) for c in label_upper if c.isdigit()), 3)
            if stars <= 2:
                return -abs(score)
            if stars == 3:
                return 0.0
            return abs(score)

        if "NEG" in label_upper or label_upper == "LABEL_0":
            return -abs(score)
        if "NEU" in label_upper:
            return 0.0
        # POSITIVE, LABEL_1, and anything else default to positive polarity.
        return abs(score)

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyse a single string."""
        text = (text or "").strip()
        if not text:
            return {"sentiment_label": "NEUTRAL", "sentiment_score": 0.0, "sentiment": 0.0}

        self._ensure_loaded()
        result = self._pipeline(text, truncation=True, max_length=self._max_length)[0]
        label = result.get("label", "NEUTRAL")
        score = float(result.get("score", 0.0))
        return {
            "sentiment_label": label,
            "sentiment_score": round(score, 6),
            "sentiment": round(self._to_signed(label, score), 6),
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Analyse many strings efficiently in batches."""
        if not texts:
            return []
        self._ensure_loaded()

        cleaned = [(t or "").strip() for t in texts]
        outputs: List[Dict[str, Any]] = [None] * len(cleaned)  # type: ignore

        # Score non-empty texts through the pipeline; empties default to neutral.
        nonempty_idx = [i for i, t in enumerate(cleaned) if t]
        nonempty_texts = [cleaned[i] for i in nonempty_idx]

        if nonempty_texts:
            raw_results = self._pipeline(
                nonempty_texts,
                truncation=True,
                max_length=self._max_length,
                batch_size=batch_size,
            )
            for pos, res in zip(nonempty_idx, raw_results):
                label = res.get("label", "NEUTRAL")
                score = float(res.get("score", 0.0))
                outputs[pos] = {
                    "sentiment_label": label,
                    "sentiment_score": round(score, 6),
                    "sentiment": round(self._to_signed(label, score), 6),
                }

        for i in range(len(outputs)):
            if outputs[i] is None:
                outputs[i] = {
                    "sentiment_label": "NEUTRAL",
                    "sentiment_score": 0.0,
                    "sentiment": 0.0,
                }
        return outputs

    # ------------------------------------------------------------------ #
    # Article-level convenience
    # ------------------------------------------------------------------ #
    def analyze_articles(
        self,
        articles: List[Dict[str, Any]],
        text_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """Annotate each article dict in-place with sentiment fields.

        Args:
            articles: List of article dicts.
            text_fn: Optional function selecting the text to analyse for an
                article.  Defaults to ``title + summary``.
            batch_size: Pipeline batch size.
        """
        if not articles:
            return articles

        if text_fn is None:
            def text_fn(a: Dict[str, Any]) -> str:  # type: ignore[misc]
                return " ".join(
                    p for p in [a.get("title", ""), a.get("body") or a.get("summary") or ""] if p
                ).strip()

        texts = [text_fn(a) for a in articles]
        scores = self.analyze_batch(texts, batch_size=batch_size)
        for article, score in zip(articles, scores):
            article.update(score)

        n_pos = sum(1 for a in articles if a.get("sentiment", 0) > 0)
        n_neg = sum(1 for a in articles if a.get("sentiment", 0) < 0)
        logger.info(
            "Scored %d articles (%d positive / %d negative).",
            len(articles),
            n_pos,
            n_neg,
        )
        return articles


if __name__ == "__main__":  # `python -m ml.sentiment_analyzer`
    from config.logging_config import configure_logging

    configure_logging()
    analyzer = SentimentAnalyzer()
    samples = [
        "Ceasefire talks collapse as violence escalates across the region.",
        "Diplomatic breakthrough brings renewed hope for lasting peace.",
    ]
    for text, res in zip(samples, analyzer.analyze_batch(samples)):
        print(f"{res['sentiment']:+.3f}  {res['sentiment_label']:<10}  {text}")
