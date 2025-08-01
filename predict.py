"""
predict.py
OCR pipeline: word segmentation + TrOCR-based recognition.
Compatible with Gradio Space + NoteBuddy OCR project.
"""

import os
import io
import time
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------- Configuration ----------------
MODEL_ID = os.getenv("MODEL_ID", "").strip()  # Hugging Face model ID (e.g. username/notebuddy-ocr)
LOCAL_MODEL_PATH = os.getenv("NOTE_BUDDY_MODEL_PATH", "./checkpoint-5080")  # fallback local path

# Image / segmentation parameters
MAX_IMAGE_DIM = int(os.getenv("NOTE_BUDDY_MAX_IMAGE_DIM", "1600"))
MIN_WORD_W = int(os.getenv("NOTE_BUDDY_MIN_WORD_W", "18"))
MIN_WORD_H = int(os.getenv("NOTE_BUDDY_MIN_WORD_H", "18"))
MAX_WORDS  = int(os.getenv("NOTE_BUDDY_MAX_WORDS", "200"))

# Morphology parameters
DILATE_RATIO = float(os.getenv("NOTE_BUDDY_DILATE_RATIO", "0.015"))  # fraction of image width
VERTICAL_EXPAND = int(os.getenv("NOTE_BUDDY_VERTICAL_EXPAND", "3"))  # kernel height

# Global cache
_CACHED = None


# ---------------- Model Loader ----------------
def _choose_model_source() -> str:
    """
    Decide which model source to use:
    1. MODEL_ID if set (Hub)
    2. Local path (checkpoint)
    """
    if MODEL_ID:
        return MODEL_ID
    if os.path.isdir(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    raise RuntimeError(
        f"No model found. Set MODEL_ID or ensure local path {LOCAL_MODEL_PATH} exists."
    )


def load_models(force: bool = False):
    """
    Load processor & model. Use global singleton cache.
    """
    global _CACHED
    if not force and _CACHED is not None:
        return _CACHED

    model_src = _choose_model_source()
    print(f"[predict.py] Loading model from: {model_src}")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(model_src)
    model.eval()
    _CACHED = (processor, model)
    return _CACHED


# ---------------- Image Helpers ----------------
def _load_and_normalize(image_bytes: bytes) -> Image.Image:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_img


def _segment_words(pil_img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Return a list of bounding boxes (x1,y1,x2,y2) approximating words using
    simple morphological grouping.
    """
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Try Otsu threshold first
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    black_ratio = (otsu > 0).mean()

    if black_ratio < 0.01 or black_ratio > 0.99:
        # Fallback to adaptive threshold
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
        )
    else:
        th = otsu

    # Dilate horizontally to merge characters into words
    dilate_w = max(3, int(pil_img.width * DILATE_RATIO))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_w, VERTICAL_EXPAND))
    dilated = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_WORD_W or h < MIN_WORD_H:
            continue
        # Avoid giant background boxes
        if w > pil_img.width * 0.95 and h > pil_img.height * 0.9:
            continue
        boxes.append((x, y, x + w, y + h))

    # Sort top-to-bottom, then left-to-right
    boxes.sort(key=lambda b: (b[1] // 30, b[0]))
    return boxes[:MAX_WORDS]


# ---------------- OCR Inference ----------------
def _batch_infer(crops: List[Image.Image], processor, model) -> List[str]:
    if not crops:
        return []
    pixel_values = processor(images=crops, return_tensors="pt", padding=True).pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def run_ocr(image_bytes: bytes, processor, model) -> Dict[str, Any]:
    """
    Run the entire OCR pipeline:
    1. Preprocess image
    2. Segment words
    3. Recognize each word
    """
    start = time.time()
    pil_img = _load_and_normalize(image_bytes)
    boxes = _segment_words(pil_img)
    img_np = np.array(pil_img)

    crops = []
    for (x1, y1, x2, y2) in boxes:
        crop = img_np[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)
        crops.append(crop_pil)

    preds = _batch_infer(crops, processor, model)
    sentence = " ".join([p for p in preds if p])

    return {
        "text": sentence,
        "words_detected": len(boxes),
        "words_predicted": len(preds),
        "latency_s": round(time.time() - start, 3),
    }
