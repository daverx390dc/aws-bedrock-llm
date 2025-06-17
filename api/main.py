from fastapi import FastAPI

from api.routes.qna import router as qa_router

app = FastAPI()

app.include_router(qa_router, prefix="/api")

