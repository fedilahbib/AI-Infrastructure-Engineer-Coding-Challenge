import os
from dotenv import load_dotenv

load_dotenv()

TRITON_URL = os.getenv("TRITON_URL")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
PRODUCT_COLLECTION = os.getenv("PRODUCT_COLLECTION")
LOG_COLLECTION = os.getenv("LOG_COLLECTION")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")