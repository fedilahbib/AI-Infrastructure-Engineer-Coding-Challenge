import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from pymongo import MongoClient
from app.vector_db import VectorDB
from app.config import MONGO_URI, DB_NAME, PRODUCT_COLLECTION, FAISS_INDEX_PATH
import faiss


# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
products = db[PRODUCT_COLLECTION]

# Mock Data
sample_products = [
    {"_id": 0, "name": "Red Shoe", "category": "Footwear", "price": 59.99},
    {"_id": 1, "name": "Blue Shirt", "category": "Apparel", "price": 29.99},
    {"_id": 2, "name": "Green Hat", "category": "Accessories", "price": 19.99}
]

# Fake embeddings
embeddings = np.random.rand(3, 512).astype(np.float32)

# Add to Mongo
products.delete_many({})
products.insert_many(sample_products)

# Add to FAISS
vector_db = VectorDB(dim=512)
vector_db.add(embeddings)
faiss.write_index(vector_db.index, FAISS_INDEX_PATH)
print("Mock data loaded into MongoDB and FAISS.")
