# from dotenv import load_dotenv, find_dotenv
# env_path = find_dotenv(filename=".env", raise_error_if_not_found=True)
# load_dotenv(env_path)
# from langchain_groq import ChatGroq
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
#
# llama_model=ChatGroq(model_name="llama-3.3-70b-versatile")
# memory=ConversationBufferMemory(return_messages=True)
# conversation = ConversationChain(
#     llm=llama_model,
#     memory = memory
# )
# result = conversation.invoke(input = "Who is the first black president of USA?")
# print(result["response"])
#--------------
from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv(filename=".env", raise_error_if_not_found=True)
load_dotenv(env_path)

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1) LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# 2) Prompt with a slot for chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "Your name is Cyko. You only answer questions related to AI."),
    MessagesPlaceholder("history"),      # <- memory goes here
    ("human", "{input}"),
])

# 3) Base chain -> plain text
chain = prompt | llm | StrOutputParser()

# 4) Simple in-memory store for histories (per session_id)
store = {}  # session_id -> ChatMessageHistory
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 5) Wrap chain with message history support
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def chat(session_id: str = "demo-session"):
    print("Chat started. Type your message and press Enter.")
    print("Commands: /reset (clear memory), /exit (quit). Ctrl+C or Ctrl+D to stop.\n")
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"/exit", "exit", "quit", ":q", "stop"}:
                print("Assistant: Bye! 👋")
                break
            if user.lower() == "/reset":
                store[session_id] = ChatMessageHistory()
                print("Assistant: Conversation history cleared.")
                continue

            answer = with_history.invoke(
                {"input": user},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"Assistant: {answer}\n")

        except KeyboardInterrupt:
            print("\nAssistant: Stopping chat. Bye!")
            break
        except EOFError:
            print("\nAssistant: EOF received. Bye!")
            break
        except Exception as e:
            print(f"Assistant: Oops, error: {e}\n")

if __name__ == "__main__":
    chat("my-session")  # change the session_id to keep separate conversations