import streamlit as st
import tempfile
from pypdf import PdfReader
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("📄 RAG Chatbot (Free Version)")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

# Read file
def read_file(file_path, file_name):
    if file_name.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

# Split text
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

if uploaded_file:
    # Save file (fix extension automatically)
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # ✅ FIXED CALL
    text = read_file(file_path, uploaded_file.name)

    st.write("📄 Text loaded")

    chunks = split_text(text)
    st.write(f"🧩 Chunks: {len(chunks)}")

    # TF-IDF embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks).toarray().astype("float32")

    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.success("Ask your question 👇")

    query = st.text_input("Ask a question")

    if query:
        query_vec = vectorizer.transform([query]).toarray().astype("float32")
        distances, indices = index.search(query_vec, k=3)

        context = " ".join([chunks[i] for i in indices[0]])

        st.write("### 🤖 Answer:")
        st.write(context)