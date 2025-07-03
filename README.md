# Product Matching Pipeline

##  Overview
This project implements an image-based product matching pipeline using:
- Vector similarity search (FAISS)
- MongoDB for metadata storage and logging
- A quantized Vision-Language Model (VLM) served via NVIDIA Triton
- FastAPI for the HTTP interface

Given an image, the system returns the closest matching product with its metadata.

---

## Architecture
```
User Image ➝ FastAPI ➝ Triton (TensorRT) ➝ Embedding ➝ FAISS + MongoDB ➝ Closest Match
```

- `app/` – Core FastAPI app (embedding, matching, vector DB, Mongo wrapper)
- `models/` – Contains Triton-compatible model (TensorRT engine)
- `scripts/` – Helper scripts to quantize model and load data
- `.env` – Environment configuration
- `docker-compose.yml` – Starts MongoDB, Triton server, and FastAPI app

---

## Getting Started

### 1. Clone the Repo
```bash
git clone <repo_url>
cd product-matching-pipeline
```

### 2. Install Requirements (if running locally)
```bash
pip install -r requirements.txt
```

### 3. Set Up the .env File
Create a `.env` in the root with:
```env
TRITON_URL=http://triton:8000
MONGO_URI=mongodb://mongo:27017
DB_NAME=productdb
PRODUCT_COLLECTION=products
LOG_COLLECTION=logs
FAISS_INDEX_PATH=./faiss_index.index
```

### 4. Prepare Data
```bash
python scripts/prepare_data.py
```
This loads:
- Mock product metadata into MongoDB
- Random image embeddings into FAISS

### 5. Build TensorRT Engine
Assumes you have `clip_vision.onnx` exported already:
```bash

# Install dependencies in virtualenv
make install

# Export to ONNX
make export

# Quantize to TensorRT engine
make quantize

# Prepare Triton layout
make triton

# Or run all steps
make all

# Clean everything
make clean

```
Output: `model.plan` inside `models/clip_vision/1`

---

## Run with Docker
```bash
docker-compose up --build
```

### Ports:
- `MongoDB`: `27017`
- `Triton Inference Server`: `8000`
- `FastAPI App`: `8080`

---

## API Usage
### Match Product by Image
```bash
curl -X POST http://localhost:8080/match \
     -F "file=@path_to_image.jpg"
```
**Response:**
```json
{
  "_id": 1,
  "name": "Blue Shirt",
  "category": "Apparel",
  "price": 29.99
}
```