from predict_torch import infer
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File

app = FastAPI() 

@app.post("/extract-table-trace") 
async def root(image: UploadFile = File(...)): 
    # contents = image.file.read()
    if image is None:
        return{"message": "Not found image"}
    image = Image.open(image.file).convert("RGB")
    image = np.array(image)
    pred_code = infer(image)
    return {"table": pred_code} 
