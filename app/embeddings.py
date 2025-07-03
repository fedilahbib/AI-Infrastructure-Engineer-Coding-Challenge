from PIL import Image
import numpy as np
import io
import httpx

async def extract_embedding(image_bytes: bytes) -> np.ndarray:
    # Load and preprocess image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    image_np = np.array(image).astype(np.float32) / 255.0

    # Normalize with CLIP mean and std
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image_np = (image_np - mean) / std

    # Convert to (1, 3, 224, 224)
    image_np = image_np.transpose(2, 0, 1)
    image_np = np.expand_dims(image_np, axis=0)

    # Build Triton V2 inference request
    payload = {
        "inputs": [
            {
                "name": "pixel_values",
                "shape": list(image_np.shape),
                "datatype": "FP32",
                "data": image_np.flatten().tolist()
            }
        ],
        "outputs": [
            {
                "name": "image_embeds"  # MUST match output name in config.pbtxt
            }
        ]
    }

    # Call Triton HTTP V2 API
    async with httpx.AsyncClient() as client:
        url = "http://localhost:8000/v2/models/clip_vision/infer"
        response = await client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        if "outputs" not in result:
            raise ValueError(f"Missing 'outputs' in Triton response: {result}")

        # Extract and return the embedding
        embedding_data = result["outputs"][0]["data"]
        return np.array(embedding_data, dtype=np.float32)
