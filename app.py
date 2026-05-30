import streamlit as st
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FRM AI Assistant", page_icon="📘", layout="wide")
st.title("📘 FRM AI Chatbot (Stable FAISS + Gemini)")

# ---------- GEMINI SETUP ----------
api_key = st.secrets.get("GEMINI_API_KEY", None)

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets")
    st.stop()

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-pro")

# ---------- LOAD FAISS (NO EMBEDDINGS NEEDED HERE) ----------
db = FAISS.load_local(
    "VectorDB",
    allow_dangerous_deserialization=True
)

# ---------- SESSION STATE ----------
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
    try:
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        context = ""
        st.warning(f"FAISS error: {e}")

    # ---------- GEMINI PROMPT ----------
    prompt = f"""
You are an expert FRM tutor.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer in structured format:
📌 Simple Explanation
📘 FRM Definition
📊 Key Points
💡 Intuition
🎯 Exam Tip

If context is insufficient, clearly say so.
"""

    # ---------- GEMINI RESPONSE ----------
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"❌ Gemini API Error: {e}"

    # ---------- OUTPUT ----------
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

    # ---------- DEBUG ----------
    with st.expander("📚 Retrieved Context"):
        st.write(context)
