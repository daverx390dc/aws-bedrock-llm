import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH')
RAW_DOCUMENTS_DIR = os.getenv('RAW_DOCUMENTS_DIR')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID=os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

if not all([VECTOR_DB_PATH, RAW_DOCUMENTS_DIR]):
    raise ValueError("Missing required environment variables: VECTOR_DB_PATH or RAW_DOCUMENTS_DIR")
