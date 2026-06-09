from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class ExactCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        key = self._key(query)
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._store), "hits": self._hits, "misses": self._misses, "hit_rate": self.hit_rate()}


class SemanticCache:
    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []
        self._hits = 0
        self._misses = 0
        self._embed_fn = DefaultEmbeddingFunction()

    def _embed(self, text: str) -> np.ndarray:
        return np.array(self._embed_fn([text])[0])

    def get(self, query: str) -> str | None:
        if not self._queries:
            self._misses += 1
            return None

        query_emb = self._embed(query)
        best_idx = -1
        best_sim = -1.0
        for i, emb in enumerate(self._embeddings):
            cos_sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-10)
            if cos_sim > best_sim:
                best_sim = cos_sim
                best_idx = i

        if best_sim >= self.threshold:
            self._hits += 1
            return self._answers[best_idx]
        self._misses += 1
        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        self._embeddings.append(self._embed(query))
        self._answers.append(answer)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._queries),
            "threshold": self.threshold,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate(),
        }
