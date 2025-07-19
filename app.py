import os
import time
from typing import Optional
import traceback
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predict import load_models, run_ocr  # our local module

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: limit PyTorch thread usage on small CPUs
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

app = FastAPI(title="NoteBuddy OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Model (lazy load) ----------
processor = model = None
MODEL_LOAD_LATENCY = None
load_error = None


@app.get("/")
def home():
    return {
        "message": "NoteBuddy OCR API running",
        "use": "POST /predict (multipart form field 'file')",
        "model_loaded": model is not None,
        "model_load_latency_s": MODEL_LOAD_LATENCY,
        "load_error": load_error,
    }


@app.get("/healthz")
def healthz():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": load_error or "model_not_loaded"}
        )
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global processor, model
    if model is None or processor is None:
        # Try one more time (lazy reload) before failing
        try:
            m_proc, m_model = load_models(force=True)
            print("Model loaded successfully.")
        except Exception as e:
            print("Error while loading model:", e)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Model not ready: {e}")
        else:
            processor, model = m_proc, m_model

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        image_bytes = await file.read()
        result = run_ocr(image_bytes, processor, model)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
