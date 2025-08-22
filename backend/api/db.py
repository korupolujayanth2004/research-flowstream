import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION = "reports"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def _make_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0, check_compatibility=False)
    else:
        return QdrantClient(url=QDRANT_URL, timeout=30.0, check_compatibility=False)

qdrant = _make_client()

def _ensure_collection():
    try:
        qdrant.get_collection(collection_name=COLLECTION)
        return
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

_ensure_collection()

def save_report(report_id: str, text: str, title: str):
    vector = embedding_model.encode(text).tolist()
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=report_id,
                vector=vector,
                payload={"text": text, "title": title},
            )
        ],
    )

def list_reports() -> List[Dict[str, Any]]:
    hits, _ = qdrant.scroll(collection_name=COLLECTION, limit=50)
    return [{"id": h.id, "title": h.payload.get("title") or "(untitled)", "text": h.payload.get("text", "")} for h in hits]

def search_reports(query: str) -> List[Dict[str, Any]]:
    vector = embedding_model.encode(query).tolist()
    hits = qdrant.search(collection_name=COLLECTION, query_vector=vector, limit=5)
    return [
        {"id": hit.id, "score": float(hit.score), "title": hit.payload.get("title") or "(untitled)", "text": hit.payload.get("text", "")}
        for hit in hits
    ]
