"""Data-access layer: Polymarket API, multi-source news scraper, and Firestore."""

from data.firebase_client import FirestoreClient
from data.news_scraper import NewsScraper
from data.polymarket_client import PolymarketClient

__all__ = ["PolymarketClient", "NewsScraper", "FirestoreClient"]
