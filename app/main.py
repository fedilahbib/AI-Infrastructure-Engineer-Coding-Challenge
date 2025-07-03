from fastapi import FastAPI, File, UploadFile
from matching import match_product

app = FastAPI()

@app.post("/match")
async def match(file: UploadFile = File(...)):
    image = await file.read()
    result = await match_product(image)
    return result