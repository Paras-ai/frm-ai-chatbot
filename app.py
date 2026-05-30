import streamlit as st
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FRM AI Assistant", page_icon="📘", layout="wide")

st.title("📘 FRM AI Chatbot (Hybrid: Books + AI)")

# ---------- GEMINI SETUP ----------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- LOAD VECTOR DB ----------
db = FAISS.load_local(
    "VectorDB",   # IMPORTANT: use relative path for cloud
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

    # ---------- STEP 1: GET CONTEXT ----------
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # ---------- STEP 2: GEMINI PROMPT ----------
    prompt = f"""
You are an expert FRM Level 2 tutor.

Use ONLY the given FRM textbook context.

Context:
{context}

Question:
{query}

Format answer:

📌 Simple Explanation
📘 FRM Definition
📊 Key Points
💡 Intuition
🎯 Exam Tip

If context is not enough, clearly say so.
"""

    response = model.generate_content(prompt)
    answer = response.text

    # ---------- SHOW ANSWER ----------
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # ---------- SOURCE ----------
    with st.expander("📚 Source Context"):
        st.write(context)
