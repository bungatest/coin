"""
Indonesian Rupiah Coin Classifier - Web Server Launcher
Run this script to start the FastAPI server and open the live scanner in your browser:
    python run.py
"""

import sys
import webbrowser
from pathlib import Path
import uvicorn

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    print("=" * 60)
    print("🪙 INDONESIAN RUPIAH COIN SCANNER (LIVE AI WEB APP)")
    print(f"🚀 Server launching at: {url}")
    print("📱 Membuka scanner di browser secara otomatis...")
    print("=" * 60)
    
    # Try to open browser after a brief delay
    try:
        import threading
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()
    except Exception:
        pass
        
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
