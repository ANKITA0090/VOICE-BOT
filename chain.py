import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import SYSTEM_PROMPT, LLM_MODEL, TEMPERATURE, MEMORY_WINDOW, DB_FOLDER, TOP_K_RESULTS, EMBED_MODEL

load_dotenv()


def _load_retriever():
    if not os.path.exists(DB_FOLDER):
        return None
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": TOP_K_RESULTS})


class VoiceBot:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
        self.retriever = _load_retriever()
        self.history: list = [SystemMessage(content=SYSTEM_PROMPT)]

        if self.retriever:
            print("RAG enabled — knowledge base loaded from ChromaDB.")
        else:
            print("RAG disabled — no database found. Run ingest.py to enable it.")

    def chat(self, user_input: str) -> str:
        # If RAG is available, retrieve context and prepend to question
        if self.retriever:
            docs = self.retriever.invoke(user_input)
            context = "\n\n".join(d.page_content for d in docs)
            augmented_input = f"Context from knowledge base:\n{context}\n\nQuestion: {user_input}"
        else:
            augmented_input = user_input

        self.history.append(HumanMessage(content=augmented_input))
        response = self.llm.invoke(self.history)
        self.history.append(AIMessage(content=response.content))

        # Keep memory within window (system prompt + last N messages)
        if len(self.history) > MEMORY_WINDOW + 1:
            self.history = [self.history[0]] + self.history[-MEMORY_WINDOW:]

        return response.content

    def clear(self):
        self.history = [SystemMessage(content=SYSTEM_PROMPT)]
