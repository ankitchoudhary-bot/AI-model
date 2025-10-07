import json
import uuid
from pathlib import Path
import re
from pypdf import PdfReader
import nltk

nltk.download('punkt')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = PROJECT_ROOT / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        text = p.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def clean_text(text: str) -> str:
    # basic cleaning — remove repeated newlines, weird whitespace
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def chunk_text_sentence_aware(text: str, min_words: int = 100, max_words: int = 300):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        w = len(sent.split())
        if current_words + w > max_words:
        # flush if we already have >= min words
            if current_words >= min_words:
                chunks.append(" ".join(current).strip())
                current = [sent]
                current_words = w
            else:
                # if too small but adding would exceed max, still add and flush
                current.append(sent)
                chunks.append(" ".join(current).strip())
                current = []
                current_words = 0
        else:
            current.append(sent)
            current_words += w


    if current:
        chunks.append(" ".join(current).strip())
    return chunks

if __name__ == "__main__":
    all_docs = list(DATA_DIR.glob("*.pdf"))
    out_file = CHUNKS_DIR / "chunks.jsonl"
    print(f"Found {len(all_docs)} PDF(s) in {DATA_DIR}")


    with out_file.open("w", encoding="utf-8") as fh:
        for doc in all_docs:
            raw = read_pdf_text(doc)
            cleaned = clean_text(raw)
            doc_chunks = chunk_text_sentence_aware(cleaned)
            print(f"Document: {doc.name} -> {len(doc_chunks)} chunks")
            for c in doc_chunks:
                rec = {"id": str(uuid.uuid4()), "source": doc.name, "text": c}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
    print(f"Saved chunks to {out_file}")