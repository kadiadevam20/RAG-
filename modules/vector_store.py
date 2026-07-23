"""
Qdrant vector store wrapper.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "knowledge_base"


@dataclass
class SearchResult:
    text: str
    source: str
    score: float


class VectorStore:
    def __init__(self, storage_path: str = "./qdrant_storage", vector_size: int = 384) -> None:
        self.client = QdrantClient(path=storage_path)
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def reset(self) -> None:
        self.client.delete_collection(collection_name=COLLECTION_NAME)
        self._ensure_collection()

    def add_documents(self, texts: List[str], sources: List[str], embeddings: np.ndarray) -> int:
        if len(texts) == 0:
            return 0
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=embeddings[i].tolist(), payload={"text": texts[i], "source": sources[i]})
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    def is_empty(self) -> bool:
        info = self.client.get_collection(collection_name=COLLECTION_NAME)
        return info.points_count == 0

    def document_count(self) -> int:
        info = self.client.get_collection(collection_name=COLLECTION_NAME)
        return info.points_count or 0

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[SearchResult]:
        response = self.client.query_points(collection_name=COLLECTION_NAME, query=query_embedding.tolist(), limit=top_k)
        return [
            SearchResult(text=p.payload.get("text", ""), source=p.payload.get("source", "unknown"), score=p.score)
            for p in response.points
        ]
