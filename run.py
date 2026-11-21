"""Generate 3D Mesh. `python run.py` opens it as a window; `python run.py web` opens it in your browser."""
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

PORT = int(os.environ.get("MESH_PORT", "8780"))
URL = f"http://localhost:{PORT}"


def _open() -> None:
    for _ in range(60):
        try:
            urllib.request.urlopen(URL, timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    mode = "web" if "web" in sys.argv[1:] else "app"
    candidates = (
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    exe = next((p for p in candidates if p and os.path.exists(p)), None)
    if mode == "app" and exe:
        subprocess.Popen([exe, f"--app={URL}", "--window-size=1120,820"])
    else:
        webbrowser.open(URL)


if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=PORT, log_level="warning")
