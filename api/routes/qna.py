from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
import os
from qa_chain import get_qa_chain




router = APIRouter()

class QueryRequest(BaseModel):
    question:str

@router.post("/query")
async def query_rag(request:QueryRequest):
    index_path = './vector_store/index/'
    if not os.path.isdir('./vector_store/index/'):
        raise HTTPException(status_code=400, details="No Index")
    

    qa_chain = get_qa_chain()

    result = qa_chain({"query": request.question})

    answer = result["result"]

    sources = []
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", None)
        if src:
            sources.append(doc)
    return {
        "answer": answer,
        "sources": sources
    }


