"""
embedder.py — Sentence embedding wrapper using OpenAI text-embedding-3-small.

Chosen model: text-embedding-3-small
  - 1536-dimensional embeddings
  - Strong retrieval quality, widely used in production RAG systems
  - No local model download required — runs via OpenAI API
  - Embeddings are L2-normalized by the API, so inner product == cosine similarity,
    which is required for correct FAISS IndexFlatIP scoring.

Requires: OPENAI_API_KEY environment variable (set in .env).
"""

from __future__ import annotations

import logging
import os
import time
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class Embedder:
    """OpenAI embedding client wrapper."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._client = None

    def _load(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Add it to your .env file."
                )
            self._client = OpenAI(api_key=api_key, max_retries=6)
            logger.info("OpenAI embedding client initialised (model: %s)", self.model_name)

    def embed(
        self,
        texts: list[str],
        batch_size: int = 100,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of document texts.
        Returns L2-normalized float32 array of shape (N, dim).
        Sends texts in batches to stay within the API's per-request limit.
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
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            # Results are returned in the same order as input
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeddings)
            time.sleep(1)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns L2-normalized float32 array of shape (1, dim).
        OpenAI handles query/document symmetry internally — no prefix needed.
        """
        self._load()
        response = self._client.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        vec = response.data[0].embedding
        return np.array([vec], dtype=np.float32)

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM
