from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from config import RAW_DOCUMENTS_DIR, VECTOR_DB_PATH
from utils import load_raw_documents, chunk_raw_documents
import os

def ingest():
    if not os.path.exists(RAW_DOCUMENTS_DIR):
        raise FileNotFoundError(f"Raw documents directory {RAW_DOCUMENTS_DIR} does not exist")

    raw_documents = load_raw_documents(RAW_DOCUMENTS_DIR)
    chunks = chunk_raw_documents(raw_documents)

    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    print(f"Index saved to {VECTOR_DB_PATH}")

if __name__ == '__main__':
    ingest()
