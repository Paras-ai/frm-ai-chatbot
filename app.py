import streamlit as st
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# ---------- CONFIG ----------
st.set_page_config(page_title="FRM AI Assistant", page_icon="📘", layout="wide")
st.title("📘 FRM AI Chatbot (Stable Version)")

# ---------- GEMINI ----------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- LOAD FAISS ----------
db = FAISS.load_local(
    "VectorDB",
    allow_dangerous_deserialization=True
)

# ---------- CHAT MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- INPUT ----------
query = st.chat_input("Ask your FRM question...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ---------- PURE FAISS TEXT SEARCH ----------
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # ---------- GEMINI PROMPT ----------
    prompt = f"""
You are an expert FRM tutor.

Use ONLY this context:

{context}

Question: {query}

Answer in:
- Simple Explanation
- FRM Definition
- Key Points
- Intuition
- Exam Tip
"""

    response = model.generate_content(prompt)
    answer = response.text

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("📚 Source Context"):
        st.write(context)
