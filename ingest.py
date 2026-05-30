"""
Run this script once to load your documents into the ChromaDB vector database.
Re-run it any time you add or change files in the data/ folder.

Usage:
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import DATA_FOLDER, DB_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL

load_dotenv()


def ingest():
    print(f"Loading documents from '{DATA_FOLDER}/'...")
    loader = DirectoryLoader(DATA_FOLDER, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"  Found {len(docs)} document(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    print("Embedding and saving to ChromaDB...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_FOLDER)
    print(f"Done. Database saved to '{DB_FOLDER}/'")
    print("You can now run: python app.py")


if __name__ == "__main__":
    ingest()
