from pymongo import MongoClient
from config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
products = db["products"]
logs = db["logs"]

def get_metadata_by_id(pid):
    return products.find_one({"_id": pid})

def log_error(error):
    logs.insert_one({"error": str(error)})

def insert_product(doc):
    products.insert_one(doc)