"""
embedder.py — Sentence embedding wrapper using Voyage AI voyage-3-lite.

Chosen model: voyage-3-lite
  - 512-dimensional embeddings
  - L2-normalized by the API — inner product == cosine similarity (correct for FAISS IndexFlatIP)
  - Retrieval-optimized: input_type="document" for indexing, input_type="query" for queries
  - Anthropic's recommended embedding provider for use with Claude

Requires: VOYAGE_API_KEY environment variable (set in .env).
"""

from __future__ import annotations

import logging
import os
import time
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "voyage-3-lite"
EMBEDDING_DIM = 512


class Embedder:
    """Voyage AI embedding client wrapper."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._client = None

    def _load(self):
        if self._client is None:
            import voyageai
            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "VOYAGE_API_KEY environment variable is not set. "
                    "Add it to your .env file."
                )
            self._client = voyageai.Client(api_key=api_key)
            logger.info("Voyage AI embedding client initialised (model: %s)", self.model_name)

    def embed(
        self,
        texts: list[str],
        batch_size: int = 128,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of document texts.
        Returns L2-normalized float32 array of shape (N, dim).
        Uses input_type="document" for asymmetric retrieval quality.
        """
        self._load()
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if show_progress:
                logger.info(
                    "Embedding batch %d/%d (%d texts) ...",
                    i // batch_size + 1,
                    (len(texts) + batch_size - 1) // batch_size,
                    len(batch),
                )
            result = self._client.embed(batch, model=self.model_name, input_type="document")
            all_embeddings.extend(result.embeddings)
            time.sleep(0.5)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns L2-normalized float32 array of shape (1, dim).
        Uses input_type="query" for asymmetric retrieval quality.
        """
        self._load()
        result = self._client.embed([text], model=self.model_name, input_type="query")
        return np.array([result.embeddings[0]], dtype=np.float32)

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM
