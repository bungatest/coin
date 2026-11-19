import json
import time
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_mobilenetv2.pt"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

class CoinClassifier:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.classes = self.metadata.get("classes", ["Rp100", "Rp1000", "Rp200", "Rp500"])
        self.classes_meta = self.metadata.get("classes_metadata", {})
        
        # Build model architecture matching trained weights
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(self.classes))
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        
        # Transforms matching evaluation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.metadata.get("mean", [0.485, 0.456, 0.406]),
                std=self.metadata.get("std", [0.229, 0.224, 0.225])
            )
        ])
        
        print(f"CoinClassifier initialized on device: {self.device}")

    def predict(self, image: Image.Image, crop_center_ratio: float = 0.7):
        """
        Runs coin classification on a PIL image.
        crop_center_ratio: if > 0 and < 1.0, crops center viewfinder region first
        """
        start_time = time.perf_counter()
        
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Optional reticle crop if full camera frame was sent
        if 0 < crop_center_ratio < 1.0:
            w, h = image.size
            crop_size = int(min(w, h) * crop_center_ratio)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))
            
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
        top_idx = int(probs.argmax())
        predicted_class = self.classes[top_idx]
        confidence = float(probs[top_idx])
        
        infer_time = (time.perf_counter() - start_time) * 1000.0
        meta = self.classes_meta.get(predicted_class, {})
        
        all_probs = {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes, probs)}
        
        return {
            "prediction": predicted_class,
            "nominal": meta.get("nominal", 0),
            "display": meta.get("display", predicted_class),
            "spoken": meta.get("spoken", predicted_class),
            "color": meta.get("color", "#6366F1"),
            "badge_color": meta.get("badge_color", "#4338CA"),
            "description": meta.get("description", ""),
            "confidence": round(confidence, 4),
            "is_confident": confidence >= 0.60,
            "probabilities": all_probs,
            "inference_time_ms": round(infer_time, 2)
        }

# Global singleton
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = CoinClassifier()
    return classifier
