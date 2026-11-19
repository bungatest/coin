import io
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

from .model_loader import get_classifier

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Indonesian Coin Scanner API",
    description="Real-time Rupiah Coin Classifier using MobileNetV2",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Base64ImageRequest(BaseModel):
    image: str
    crop_center_ratio: float = 0.75  # crop center ratio (matches viewfinder reticle)

@app.on_event("startup")
def startup_event():
    print("Preloading MobileNetV2 model...")
    get_classifier()
    print("Model ready!")

@app.get("/api/health")
def health():
    clf = get_classifier()
    return {
        "status": "online",
        "model": clf.metadata.get("model_name", "mobilenet_v2"),
        "classes": clf.classes,
        "device": str(clf.device)
    }

@app.get("/api/metadata")
def metadata():
    clf = get_classifier()
    return clf.metadata

@app.post("/api/predict")
async def predict_file(
    file: UploadFile = File(None),
    crop_center_ratio: float = 0.75
):
    """
    Accepts multipart file upload (image)
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        clf = get_classifier()
        result = clf.predict(image, crop_center_ratio=crop_center_ratio)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/predict-base64")
async def predict_base64(payload: Base64ImageRequest):
    """
    Accepts base64 encoded image string (from webcam canvas)
    """
    try:
        raw_b64 = payload.image
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
        image = Image.open(io.BytesIO(image_bytes))
        
        clf = get_classifier()
        result = clf.predict(image, crop_center_ratio=payload.crop_center_ratio)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 inference error: {str(e)}")

# Mount frontend directory for static serving
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Frontend index.html not yet found"}
