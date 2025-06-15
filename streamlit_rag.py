import streamlit as st
from io import BytesIO
import torch

import PyPDF2
from pdfminer.high_level import extract_text as pdfminer_extract_text

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- PDF Text Extraction ---
def extract_text_pypdf2(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception:
        return None

def extract_text_pdfminer(file):
    try:
        file.seek(0)
        return pdfminer_extract_text(file)
    except Exception:
        return ""

def extract_text(file):
    text = extract_text_pypdf2(file)
    if not text or text.strip() == "":
        text = extract_text_pdfminer(file)
    return text

# --- Caching Embedding Model ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Caching LaMini-GPT Model ---
@st.cache_resource
def load_lamini_gpt():
    tokenizer = AutoTokenizer.from_pretrained(
        "MBZUAI/LaMini-GPT-774M", use_auth_token=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "MBZUAI/LaMini-GPT-774M", use_auth_token=True
    )
    device = 0 if torch.cuda.is_available() else -1
    nlp = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device
    )
    return nlp


# @st.cache_resource
# def load_qwen2_1b():
#     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1B", use_auth_token=True)
#     model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1B", use_auth_token=True)
#     device = 0 if torch.cuda.is_available() else -1
#     nlp = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
#     return nlp

# --- Streamlit App ---
st.title("PDF Answer Engine")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    st.spinner("Extracting text from PDF...")
    text = extract_text(uploaded_file)
    if not text:
        st.error("Failed to extract text from PDF.")
        st.stop()

    # Split text into chunks
    chunk_size = 500
    overlap = 100
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])

    # Embedding model (sentence-transformers)
    st.spinner("Embedding PDF chunks for semantic search...")
    embedder = load_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Load LaMini-GPT model
    st.spinner("Loading LaMini-GPT-774M model...")
    nlp = load_lamini_gpt()

    st.success("Ready! Ask a question about your PDF below.")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        # Embed the query
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embedding, k=3)
        retrieved = "\n".join([chunks[i] for i in I[0]])

        # Build prompt and get answer
        prompt = f"Context:\n{retrieved}\n\nQuestion: {query}\nAnswer:"
        response = nlp(prompt, max_new_tokens=128, do_sample=False)[0]['generated_text']
        # Post-process to get only the answer (optional)
        answer = response.split("Answer:")[-1].strip()
        st.write("**Answer:**")
        st.write(answer)