"""FinBERT sentiment pipeline for financial text.

Extracted from notebooks/03_finbert_sentiment.ipynb.
Uses ProsusAI/finbert as a frozen feature extractor — no fine-tuning.
"""

import logging
from dataclasses import dataclass

import pandas as pd
from transformers import pipeline

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment scores for a single headline."""

    headline: str
    positive: float
    negative: float
    neutral: float
    net_sentiment: float  # positive - negative, range [-1, 1]


class FinBERTSentiment:
    """FinBERT sentiment pipeline wrapper.

    Usage:
        sentiment = FinBERTSentiment()
        results = sentiment.score(["Fed raises rates by 75bps"])
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        logger.info("Loading FinBERT from %s", model_name)
        self._pipeline = pipeline(
            "sentiment-analysis", model=model_name, top_k=None
        )
        logger.info("FinBERT loaded")

    def score(self, headlines: list[str]) -> list[SentimentResult]:
        """Score a list of headlines. Returns SentimentResult per headline."""
        raw = self._pipeline(headlines, batch_size=8)
        results = []
        for headline, scores in zip(headlines, raw):
            score_dict = {s["label"]: s["score"] for s in scores}
            results.append(
                SentimentResult(
                    headline=headline,
                    positive=score_dict.get("positive", 0),
                    negative=score_dict.get("negative", 0),
                    neutral=score_dict.get("neutral", 0),
                    net_sentiment=(
                        score_dict.get("positive", 0)
                        - score_dict.get("negative", 0)
                    ),
                )
            )
        return results

    def score_df(self, headlines: list[str]) -> pd.DataFrame:
        """Score headlines and return as a DataFrame."""
        results = self.score(headlines)
        return pd.DataFrame([vars(r) for r in results])
