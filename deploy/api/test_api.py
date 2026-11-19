"""
MLOps Automated Integration Test Suite for Rupiah Coin Scanner API
Runs tests against FastAPI backend:
    python test_api.py
"""

import sys
import time
import base64
import io
import asyncio
from pathlib import Path
from PIL import Image, ImageDraw
import httpx

# Add deploy root to sys.path
DEPLOY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEPLOY_ROOT))

from backend.main import app

def create_synthetic_coin_image():
    """Generates a synthetic coin image for testing."""
    img = Image.new("RGB", (300, 300), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw circular coin
    draw.ellipse((50, 50, 250, 250), fill=(200, 200, 210), outline=(140, 140, 150), width=4)
    # Inner ring
    draw.ellipse((75, 75, 225, 225), outline=(160, 160, 170), width=2)
    return img

async def run_async_tests():
    print("=" * 60)
    print("🧪 RUNNING MLOps INTEGRATION TESTS")
    print("=" * 60)
    
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check test
        print("\n[TEST 1] Checking GET /api/health ...")
        res = await client.get("/api/health")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        health_data = res.json()
        assert health_data["status"] == "online"
        assert "mobilenet" in health_data["model"].lower()
        print(f"  ✅ PASS: Status={health_data['status']}, Model={health_data['model']}, Device={health_data['device']}")
        
        # 2. Metadata check test
        print("\n[TEST 2] Checking GET /api/metadata ...")
        res = await client.get("/api/metadata")
        assert res.status_code == 200
        meta = res.json()
        assert len(meta["classes"]) == 4
        print(f"  ✅ PASS: Classes detected: {meta['classes']}")
        
        # 3. Multipart predict test
        print("\n[TEST 3] Testing POST /api/predict (Multipart Form) ...")
        test_img = create_synthetic_coin_image()
        img_byte_arr = io.BytesIO()
        test_img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        t0 = time.perf_counter()
        res = await client.post(
            "/api/predict",
            files={"file": ("test_coin.jpg", img_bytes, "image/jpeg")}
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        assert res.status_code == 200, f"Error: {res.text}"
        pred_data = res.json()
        assert "prediction" in pred_data
        assert "confidence" in pred_data
        assert "probabilities" in pred_data
        print(f"  ✅ PASS: Prediction={pred_data['prediction']}, Conf={pred_data['confidence']:.4f}, Latency={latency_ms:.2f}ms")
        
        # 4. Base64 predict test
        print("\n[TEST 4] Testing POST /api/predict-base64 (Webcam Payload) ...")
        b64_str = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")
        t0 = time.perf_counter()
        res = await client.post(
            "/api/predict-base64",
            json={"image": b64_str, "crop_center_ratio": 0.8}
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        assert res.status_code == 200
        b64_pred = res.json()
        assert b64_pred["prediction"] in meta["classes"]
        print(f"  ✅ PASS: Base64 Prediction={b64_pred['prediction']}, Latency={latency_ms:.2f}ms")
        
        # 5. Frontend index route test
        print("\n[TEST 5] Testing GET / (Static Web App) ...")
        res = await client.get("/")
        assert res.status_code == 200
        assert "Rupiah Coin Scanner" in res.text
        print("  ✅ PASS: Frontend HTML served successfully!")
        
    print("\n" + "=" * 60)
    print("🎉 ALL MLOps INTEGRATION TESTS PASSED (5/5)!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_async_tests())
