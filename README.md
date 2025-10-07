1. Create a virtual environment and install dependencies:

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

2. Place your PDF(s) inside the `data/` folder.

3. Run the **chunking** step to extract and split text:

```bash
python src/ingest.py
```

4. Create embeddings and build the FAISS index:

```bash
python src/embeddings.py
```

5. Launch the Streamlit chatbot application:

```bash
streamlit run app.py
```

6. Open the provided local URL in your browser (default: [http://localhost:8501](http://localhost:8501)) and start chatting with your documents.

---
7. Example queries to try

Once your document(s) are indexed and the app is running, try these queries to validate the chatbot:

* "Summarize the main objectives described in this document."
* "According to the text, what are the steps in the RAG pipeline?"
* "What does the chunking strategy mention about sentence limits?"
* "Explain how embeddings are used in this chatbot."
* "If the answer is not found in the document, how should the assistant respond?"

---

8. Notes & suggestions

Model choice: The default `gpt2` model is for demonstration only. For real RAG tasks, replace it with an instruction-tuned model (e.g., `mistral-7b-instruct`, `llama-3-8b-instruct`, or `zephyr-7b-alpha`).
Streaming support: The app uses Hugging Face `TextIteratorStreamer` for token-by-token generation. Larger models may require GPU and quantization for efficient inference.
Retrieval tuning: Modify `top_k` in the `retriever.search()` call (default 5) to adjust how many chunks are retrieved.
Persistence: Generated embeddings and FAISS index are saved in `/vectordb`. You can reuse them without re-running `embeddings.py` each time.
Future improvements: Add caching, use a cross-encoder for re-ranking, or integrate external vector databases like Chroma or Qdrant.

