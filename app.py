import streamlit as st
from src.retriever import Retriever
from src.generator import build_prompt, stream_generate

MODEL_NAME = "gpt2"

st.set_page_config(page_title="RAG Chatbot (streaming)", layout="wide")
st.title("RAG Chatbot — Streaming Demo")

# Sidebar info
st.sidebar.header("Configuration")
st.sidebar.write("Model: ")
st.sidebar.write(MODEL_NAME)

if "retriever" not in st.session_state:
    with st.spinner("Loading retriever and index (this may take a few seconds)..."):
        st.session_state.retriever = Retriever()

# show number of chunks
try:
    n_chunks = len(st.session_state.retriever.metadata)
except Exception:
    n_chunks = "?"
st.sidebar.write(f"Indexed chunks: {n_chunks}")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([3, 1])

with col1:
    chat_box = st.empty()

    def render_chat():
        for msg in st.session_state.messages:
            if msg[0] == "user":
                st.markdown(f"**You:** {msg[1]}")
            else:
                st.markdown(f"**Assistant:** {msg[1]}")

    render_chat()

    user_input = st.text_input("Ask a question:")

    if st.button("Send") and user_input.strip():
        st.session_state.messages.append(("user", user_input))
        chat_box.empty()
        render_chat()

        retriever = st.session_state.retriever
        retrieved = retriever.search(user_input, top_k=5)

        prompt = build_prompt(retrieved, user_input)

        placeholder = st.empty()
        full_response = ""
        with st.spinner("Generating (streaming)..."):
            for part in stream_generate(MODEL_NAME, prompt, max_new_tokens=256):
                full_response += part
                placeholder.markdown(f"**Assistant:** {full_response}")

        st.session_state.messages.append(("assistant", full_response))

        with st.expander("Source chunks used"):
            for r in retrieved:
                st.write(f"- source: {r['source']} | score: {r['score']:.4f}")
                st.write(r['text'])

        chat_box.empty()
        render_chat()

with col2:
    st.header("Controls")
    if st.button("Reset chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.write("Tips:")
    st.write("- Use precise questions; the assistant only uses provided documents.")