import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FRM AI Assistant", page_icon="📘", layout="wide")

st.title("📘 FRM AI Chatbot")

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

    # ---------- CLOUD SAFE CONTEXT (FAISS REMOVED) ----------
    context = """
FRM Knowledge Base is currently running in cloud-safe mode.

Full vector database search is available only in local version.

We will upgrade this to Gemini-powered AI in next step.
"""

    # ---------- CHATGPT STYLE ANSWER ENGINE ----------
    answer = f"""
You are an expert FRM Level 2 tutor.

Use structured exam-style explanation.

Context:
{context}

Question:
{query}

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
