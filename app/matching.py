from embeddings import extract_embedding
from vector_db import VectorDB
from mongo_store import get_metadata_by_id, log_error

# Initialize FAISS index (assuming 512-dim vectors — adjust as needed)
vector_db = VectorDB(dim=512)

async def match_product(image_bytes):
    try:
        vec = await extract_embedding(image_bytes)
        idx = vector_db.search(vec.reshape(1, -1))[1][0][0]
        product = get_metadata_by_id(idx)
        return product
    except Exception as e:
        log_error(e)
        return None