import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
EMBED_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, index_path: Path = None, metadata_path: Path = None, model_name: str = EMBED_MODEL):
        index_path = index_path or (VECTORDB_DIR / "index.faiss")
        metadata_path = metadata_path or (VECTORDB_DIR / "metadata.json")

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as fh:
            self.metadata = json.load(fh)

        self.embedder = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            md = self.metadata[idx]
            results.append({"score": float(score), "id": md["id"], "source": md["source"], "text": md["text"]})
        return results


if __name__ == "__main__":
    r = Retriever()
    q = input("Query: ")
    res = r.search(q, top_k=3)
    for r in res:
        print(r["score"], r["source"], r["text"][:180].replace("\n", " "))