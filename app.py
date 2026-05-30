import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- UI ----------
st.set_page_config(page_title="FRM AI Chatbot", page_icon="📘")
st.title("📘 FRM AI Chatbot")

# ---------- GEMINI ----------
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- SAFE EMBEDDINGS ----------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ---------- LOAD FAISS (IMPORTANT FIX) ----------
@st.cache_resource
def load_db():
    return FAISS.load_local(
        "VectorDB",
        embeddings,
        allow_dangerous_deserialization=True
    )

db = load_db()

# ---------- CHAT MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- INPUT ----------
query = st.chat_input("Ask FRM question...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ---------- RETRIEVE ----------
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # ---------- PROMPT ----------
    prompt = f"""
You are an FRM Level 2 expert tutor.

Use ONLY this context:

{context}

Question: {query}

Answer in:
- Simple Explanation
- Definition
- Key Points
- Intuition
- Exam Tip
"""

    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Gemini error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("Source Context"):
        st.write(context)
