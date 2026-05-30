import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FRM AI Assistant", page_icon="📘", layout="wide")
st.title("📘 FRM AI Chatbot (Stable + FAISS + Gemini)")

# ---------- GEMINI SETUP ----------
api_key = st.secrets.get("GEMINI_API_KEY", None)

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets")
    st.stop()

genai.configure(api_key=api_key)

# ⚠️ safer model (works in most accounts)
model = genai.GenerativeModel("gemini-1.5-pro")

# ---------- EMBEDDINGS (LIGHTWEIGHT SAFE) ----------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding = load_embeddings()

# ---------- LOAD FAISS ----------
@st.cache_resource
def load_db():
    return FAISS.load_local(
        "VectorDB",
        embedding,
        allow_dangerous_deserialization=True
    )

db = load_db()

# ---------- SESSION MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- INPUT ----------
query = st.chat_input("Ask your FRM question...")

if query:

    # user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ---------- FAISS SEARCH ----------
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # ---------- PROMPT ----------
    prompt = f"""
You are an expert FRM tutor.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer in structured format:
- Simple Explanation
- FRM Definition
- Key Points
- Intuition
- Exam Tip

If context is not enough, say clearly.
"""

    # ---------- GEMINI RESPONSE ----------
    response = model.generate_content(prompt)
    answer = response.text

    # save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

    # ---------- SOURCE ----------
    with st.expander("📚 FAISS Retrieved Context"):
        st.write(context)
