"""
Flask API Launcher for Postman Testing & Web Scanner
Jalankan script ini untuk menyalakan Flask API:
    python run_flask.py
"""

import sys
import webbrowser
import threading
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from backend.app_flask import app

if __name__ == "__main__":
    port = 5000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    print("=" * 65)
    print("🪙 INDONESIAN RUPIAH COIN SCANNER - FLASK API SERVER")
    print(f"🚀 Server aktif di: {url}")
    print(f"📬 Endpoint Testing Postman: POST {url}/predict")
    print("   (Kirim via Body -> form-data -> key: 'file' [File])")
    print("=" * 65)

    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)
        
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host=host, port=port, debug=False)
