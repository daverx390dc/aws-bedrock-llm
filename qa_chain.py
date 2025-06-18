from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.llms.bedrock import BedrockLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from config import VECTOR_DB_PATH
from utils import load_raw_documents, chunk_raw_documents

def get_qa_chain():
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    prompt = "Read the document and answer the quesion asked."


    llm = BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0",
                     model_kwargs = {
        "prompt": prompt,
        "temperature": 0.3,  # ✅ spelled correctly
        "top_p": 0.95,
        "max_gen_len": 512
        }
                                                 )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type='stuff'
    )

    return qa_chain
