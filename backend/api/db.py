import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION = "reports"

# --- Model Loading (from local files within the Docker image) ---
# This path corresponds to where the Dockerfile copies the model.
MODEL_PATH = "./models/all-MiniLM-L6-v2" 

# Initialize the embedding model from the local path.
try:
    print(f"Loading sentence-transformer model from local path: {MODEL_PATH}")
    embedding_model = SentenceTransformer(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ FATAL: Could not load the embedding model from {MODEL_PATH}.")
    print("This indicates an issue with the Docker build or the file path in db.py.")
    raise e

# --- Qdrant Client and Collection Setup ---
def _make_client() -> QdrantClient:
    """Creates a Qdrant client based on environment variables."""
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0, check_compatibility=False)
    else:
        return QdrantClient(url=QDRANT_URL, timeout=30.0, check_compatibility=False)

qdrant = _make_client()

def _ensure_collection():
    """Ensures the Qdrant collection exists, creating it if necessary."""
    try:
        qdrant.get_collection(collection_name=COLLECTION)
        print(f"✅ Collection '{COLLECTION}' already exists.")
    except Exception:
        print(f"🔧 Collection '{COLLECTION}' not found. Creating it...")
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print("✅ Collection created.")

_ensure_collection()

# --- Database Functions ---
def save_report(report_id: str, text: str, title: str):
    """Encodes and saves a report to Qdrant."""
    vector = embedding_model.encode(text).tolist()
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=report_id, vector=vector, payload={"text": text, "title": title})],
    )

def list_reports() -> List[Dict[str, Any]]:
    """Lists recent reports, including their titles."""
    hits, _ = qdrant.scroll(collection_name=COLLECTION, limit=50)
    return [{"id": h.id, "title": h.payload.get("title", "(untitled)"), "text": h.payload.get("text", "")} for h in hits]

def search_reports(query: str) -> List[Dict[str, Any]]:
    """Performs semantic search and returns reports with titles."""
    vector = embedding_model.encode(query).tolist()
    hits = qdrant.search(collection_name=COLLECTION, query_vector=vector, limit=5)
    return [{"id": hit.id, "score": float(hit.score), "title": hit.payload.get("title", "(untitled)"), "text": hit.payload.get("text", "")} for hit in hits]
