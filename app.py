import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FRM AI Assistant", page_icon="📘", layout="wide")

st.title("📘 FRM AI Chatbot (Hybrid: Books + AI)")

# ---------- GEMINI SETUP ----------
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- EMBEDDINGS ----------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- LOAD VECTOR DB ----------
db = FAISS.load_local(
    "C:/Users/paras/frm_ai_app/VectorDB",
    embedding,
    allow_dangerous_deserialization=True
)

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

    # ---------- STEP 1: GET FRM CONTEXT FROM BOOKS ----------
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # ---------- STEP 2: GEMINI PROMPT ----------
    prompt = f"""
You are an expert FRM Level 2 tutor.

You MUST answer ONLY using the given FRM textbook context.

Context:
{context}

Question:
{query}

Format your answer like an exam-ready response:

📌 Simple Explanation
📘 FRM Definition
📊 Key Points
💡 Intuition
🎯 Exam Tip

If context is insufficient, say so clearly.
"""

    response = model.generate_content(prompt)
    answer = response.text

    # ---------- SHOW ANSWER ----------
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # ---------- SOURCE VIEW ----------
    with st.expander("📚 Source Context (from FRM books)"):
        st.write(context)
