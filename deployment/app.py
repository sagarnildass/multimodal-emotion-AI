from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import tempfile
from inference import model_fn, predict_fn  # Reuse your code as a module

app = FastAPI()

# Load model once at startup
MODEL_DIR = "model_normalized"
model_dict = model_fn(MODEL_DIR)

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    # Validate file extension
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported")

    # Save to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_path = tmp.name
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    try:
        input_data = {"video_path": tmp_path}
        predictions = predict_fn(input_data, model_dict)
        return JSONResponse(content=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
