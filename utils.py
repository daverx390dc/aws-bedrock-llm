from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os

def load_raw_documents(raw_document_dir:str):
    all_docs = []
    for fname in os.listdir(raw_document_dir):
        file_path = os.path.join(raw_document_dir, fname)
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            print(f"Loaded file: {fname}, {len(docs)} pages")
            for doc in docs:
                doc.metadata["source"] = fname
            all_docs.extend(docs)
    return all_docs

def chunk_raw_documents(raw_documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(raw_documents)