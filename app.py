import os
import io
import time
import gradio as gr
import torch

# Import your existing helpers
from predict import load_models, run_ocr

# ------------------------------------------------------------------
# (Optional) Set this env var in the Space settings if you want to
# force using a HF Hub model id instead of local folder:
#   MODEL_ID = "<username>/<model-repo>"
#
# In predict.py you can modify _choose_model_source() to check MODEL_ID too.
# ------------------------------------------------------------------

# Load (processor, model) once. HF Spaces have enough RAM, so no lazy load.
print("[INFO] Loading model...")
load_start = time.time()
processor, model = load_models(force=False)
load_time = round(time.time() - load_start, 2)
print(f"[INFO] Model loaded in {load_time} s")

# For safety (TrOCR short sequences): cache off can slightly reduce memory.
if hasattr(model, "config"):
    try:
        model.config.use_cache = False
    except Exception:
        pass

# ------------------------------------------------------------------
# Wrapper function for Gradio
# ------------------------------------------------------------------
def ocr_image(pil_image):
    """
    Gradio callback: receives a PIL Image.
    Converts to bytes (PNG) and calls run_ocr (which expects bytes).
    """
    if pil_image is None:
        return "", f"No image provided."

    # Convert PIL image to bytes
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    try:
        result = run_ocr(image_bytes, processor, model)
    except Exception as e:
        return "", f"Error: {e}"

    text = result.get("text", "")
    info = (
        f"Words detected: {result.get('words_detected')} | "
        f"Words predicted: {result.get('words_predicted')} | "
        f"Latency: {result.get('latency_s')} s"
    )
    return text, info


# ------------------------------------------------------------------
# Gradio Interface
# ------------------------------------------------------------------
DESCRIPTION = """
**NoteBuddy OCR (Handwritten TrOCR)**  
Upload a handwritten note image to extract text.  
"""

demo = gr.Interface(
    fn=ocr_image,
    inputs=gr.Image(type="pil", label="Handwritten Note Image"),
    outputs=[
        gr.Textbox(label="Transcribed Text", lines=10),
        gr.Textbox(label="Details / Stats")
    ],
    title="NoteBuddy OCR",
    description=DESCRIPTION,
    examples=None,
    allow_flagging="never"
)

# Enable queuing so multiple users can send jobs (optional)
demo.queue(max_size=12, concurrency_count=1)

if __name__ == "__main__":
    demo.launch()
