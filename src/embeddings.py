import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_ROOT / "chunks" / "chunks.jsonl"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
VECTORDB_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            chunks.append(json.loads(ln))
    return chunks


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_FILE)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(EMBED_MODEL)
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # normalize vectors for cosine-sim via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(VECTORDB_DIR / "index.faiss"))

    # save metadata in same order as vectors
    meta = [{"id": c["id"], "source": c.get("source", ""), "text": c["text"]} for c in chunks]
    with (VECTORDB_DIR / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)

    print(f"Saved FAISS index and metadata to {VECTORDB_DIR}")