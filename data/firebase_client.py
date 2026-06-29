"""Firebase Firestore data-access layer.

Wraps ``firebase-admin`` / ``google-cloud-firestore`` to provide idempotent
upserts and simple reads for the three core collections:

* ``bbc_articles``            — BBC News articles + sentiment.
* ``polymarket_markets``      — markets + embedded probability history.
* ``polymarket_price_history``— (optional) flattened per-point series.
* ``pipeline_runs``           — run metadata for observability.

Credentials are resolved from configuration in this order:

1. ``FIREBASE_CREDENTIALS_JSON`` — inline service-account JSON.
2. ``GOOGLE_APPLICATION_CREDENTIALS`` — path to a service-account key file.

Documents are written with ``merge=True`` so re-running the pipeline updates
existing records instead of duplicating them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from config.logging_config import get_logger
from config.settings import Settings, get_settings

logger = get_logger(__name__)

# Firestore caps batched writes at 500 operations.
_BATCH_LIMIT = 450


class FirestoreClientError(RuntimeError):
    """Raised when Firestore cannot be initialised or used."""


class FirestoreClient:
    """High-level CRUD wrapper around Firestore."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._db = None  # lazily initialised
        self._init_app()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _build_credentials(self):
        """Return a firebase_admin credential object from settings."""
        from firebase_admin import credentials  # local import keeps import-time light

        if self.settings.firebase_credentials_json:
            try:
                info = json.loads(self.settings.firebase_credentials_json)
            except json.JSONDecodeError as exc:
                raise FirestoreClientError(
                    "FIREBASE_CREDENTIALS_JSON is not valid JSON."
                ) from exc
            return credentials.Certificate(info)

        if self.settings.google_application_credentials:
            return credentials.Certificate(self.settings.google_application_credentials)

        raise FirestoreClientError(
            "No Firebase credentials configured. Set FIREBASE_CREDENTIALS_JSON or "
            "GOOGLE_APPLICATION_CREDENTIALS (see .env.template)."
        )

    def _init_app(self) -> None:
        try:
            import firebase_admin
            from firebase_admin import firestore
        except ImportError as exc:  # pragma: no cover
            raise FirestoreClientError(
                "firebase-admin is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        # Reuse the default app if it already exists (Streamlit re-runs, tests).
        if not firebase_admin._apps:
            cred = self._build_credentials()
            options: Dict[str, Any] = {}
            if self.settings.firebase_project_id:
                options["projectId"] = self.settings.firebase_project_id
            firebase_admin.initialize_app(cred, options or None)
            logger.info("Initialised Firebase app (project=%s).", self.settings.firebase_project_id)
        else:
            logger.debug("Reusing existing Firebase app.")

        self._db = firestore.client()

    @property
    def db(self):
        if self._db is None:  # pragma: no cover - defensive
            self._init_app()
        return self._db

    # ------------------------------------------------------------------ #
    # Generic upsert
    # ------------------------------------------------------------------ #
    def upsert_documents(
        self,
        collection: str,
        documents: Iterable[Dict[str, Any]],
        id_field: str,
    ) -> int:
        """Batch-upsert ``documents`` into ``collection`` keyed by ``id_field``.

        Returns the number of documents written.
        """
        documents = list(documents)
        if not documents:
            logger.info("No documents to upsert into '%s'.", collection)
            return 0

        col_ref = self.db.collection(collection)
        written = 0
        batch = self.db.batch()
        ops = 0

        for doc in documents:
            doc_id = str(doc.get(id_field))
            if not doc_id or doc_id == "None":
                logger.warning("Skipping document without '%s' in '%s'.", id_field, collection)
                continue
            payload = dict(doc)
            payload["_updated_at"] = datetime.now(timezone.utc).isoformat()
            batch.set(col_ref.document(doc_id), payload, merge=True)
            ops += 1
            written += 1
            if ops >= _BATCH_LIMIT:
                batch.commit()
                batch = self.db.batch()
                ops = 0

        if ops:
            batch.commit()

        logger.info("Upserted %d documents into '%s'.", written, collection)
        return written

    # ------------------------------------------------------------------ #
    # Domain-specific helpers
    # ------------------------------------------------------------------ #
    def save_articles(self, articles: List[Dict[str, Any]]) -> int:
        return self.upsert_documents(
            self.settings.firestore_articles_collection, articles, id_field="article_id"
        )

    def save_markets(self, markets: List[Dict[str, Any]]) -> int:
        return self.upsert_documents(
            self.settings.firestore_markets_collection, markets, id_field="market_id"
        )

    def save_run(self, run_metadata: Dict[str, Any]) -> str:
        """Persist a pipeline-run summary; returns the generated document id."""
        run_id = run_metadata.get("run_id") or datetime.now(timezone.utc).strftime(
            "run_%Y%m%dT%H%M%SZ"
        )
        payload = dict(run_metadata)
        payload["run_id"] = run_id
        payload["_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.db.collection(self.settings.firestore_runs_collection).document(run_id).set(payload)
        logger.info("Saved pipeline run '%s'.", run_id)
        return run_id

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def _read_collection(self, collection: str, limit: Optional[int]) -> List[Dict[str, Any]]:
        query = self.db.collection(collection)
        if limit:
            query = query.limit(limit)
        return [doc.to_dict() for doc in query.stream()]

    def get_articles(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self._read_collection(self.settings.firestore_articles_collection, limit)

    def get_markets(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self._read_collection(self.settings.firestore_markets_collection, limit)

    # ------------------------------------------------------------------ #
    def healthcheck(self) -> bool:
        """Lightweight connectivity check; returns True if Firestore responds."""
        try:
            next(self.db.collections(), None)
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("Firestore healthcheck failed: %s", exc)
            return False


if __name__ == "__main__":  # `python -m data.firebase_client`
    from config.logging_config import configure_logging

    configure_logging()
    client = FirestoreClient()
    print("Firestore reachable:", client.healthcheck())
