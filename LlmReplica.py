import os
from uuid import uuid4

# --- Environment (.env or Streamlit Secrets) ---
from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv(filename=".env", raise_error_if_not_found=False)
if env_path:
    load_dotenv(env_path)

import streamlit as st
if "GROQ_API_KEY" not in os.environ:
    # Pull from Streamlit Secrets if available
    key = st.secrets.get("GROQ_API_KEY", None)
    if key:
        os.environ["GROQ_API_KEY"] = key

# --- LangChain / Groq imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Version-agnostic import for in-memory history
try:
    # Newer LangChain
    from langchain_core.chat_history import InMemoryChatMessageHistory
except Exception:
    # Older/community split
    from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory


# ---------- Page setup ----------
st.set_page_config(page_title="Groq + LangChain Chat", page_icon="💬", layout="centered")
st.title("💬 Chat with Memory (Groq + LangChain)")

# --- Guard: API key required ---
if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
    st.error("GROQ_API_KEY not found. Set it in your environment, .env, or Streamlit Secrets.")
    st.stop()

# ---------- Sidebar (settings) ----------
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    model_name = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile"],  # add more Groq models here if you like
        index=0,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat"):
            if "session_id" in st.session_state:
                sid = st.session_state["session_id"]
                store = st.session_state.get("store", {})
                if sid in store:
                    store[sid] = InMemoryChatMessageHistory()
                    st.session_state["store"] = store
            st.rerun()
    with col2:
        if st.button("New session"):
            st.session_state["session_id"] = str(uuid4())
            st.rerun()

# ---------- Session state ----------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid4())
if "store" not in st.session_state:
    st.session_state["store"] = {}  # session_id -> InMemoryChatMessageHistory

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    store = st.session_state["store"]
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        st.session_state["store"] = store
    return store[session_id]

# ---------- LLM + chain ----------
llm = ChatGroq(model_name=model_name, temperature=temperature)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),  # memory slot
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

session_id = st.session_state["session_id"]

# ---------- Show history ----------
history = get_session_history(session_id).messages
for m in history:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(m.content)

# ---------- Chat input ----------
user_text = st.chat_input("Type your message…")
if user_text:
    # Show the user's message immediately
    with st.chat_message("user"):
        st.write(user_text)

    # Invoke chain (this appends to session history internally)
    answer = with_history.invoke(
        {"input": user_text},
        config={"configurable": {"session_id": session_id}},
    )

    # Display assistant reply
    with st.chat_message("assistant"):
        st.write(answer)
            print(f"Assistant: Oops, error: {e}\n")

if __name__ == "__main__":
    chat("my-session")  # change the session_id to keep separate conversations
