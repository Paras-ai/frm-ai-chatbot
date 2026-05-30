import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FRM AI Assistant", page_icon="📘", layout="wide")

st.title("📘 FRM AI Chatbot")

# ---------- LOAD MODEL ----------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "C:/Users/paras/frm_ai_app/VectorDB",
    embedding,
    allow_dangerous_deserialization=True
)

# ---------- SESSION MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT HISTORY ----------
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

    # retrieve context
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # ---------- CHATGPT STYLE ANSWER ENGINE ----------
    answer = f"""
You are an expert FRM Level 2 tutor.

Use the context below to answer in a structured, exam-ready format.

Context:
{context}

Question:
{query}

Now respond in this format:

📌 Simple Explanation:
Explain in very easy language.

📘 FRM Definition:
Give proper exam definition.

📊 Key Points:
- Point 1
- Point 2
- Point 3

💡 Intuition:
Real-world understanding.

🎯 Exam Tip:
How this is asked in FRM exams.
"""

    # assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # optional expander
    with st.expander("📚 Source Context"):
        st.write(context)