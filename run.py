"""Generate 3D Mesh. `python run.py` opens it as a window; `python run.py web` opens it in your browser."""
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

def _free_port(start: int) -> int:
    """The first free port from `start` up: a second copy (or another program) holding 8780 must not stop the app from starting."""
    import socket

    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:  # nothing is listening there
                return p
    return start


PORT = _free_port(int(os.environ.get("MESH_PORT", "8780")))
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
        subprocess.Popen([exe, f"--app={URL}", "--window-size=1120,760"])
    else:
        webbrowser.open(URL)


if __name__ == "__main__":
    # Started without a console (the .exe, or pythonw from the launcher): stdout/stderr are None, which breaks uvicorn's logging. Send them to a log
    # file next to the program instead, so an error is not lost silently.
    if sys.stdout is None or sys.stderr is None:
        _here = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
        try:
            _log = open(os.path.join(_here, "mesh-app.log"), "a", encoding="utf-8", buffering=1)
        except OSError:
            _log = open(os.devnull, "w")
        sys.stdout = sys.stdout or _log
        sys.stderr = sys.stderr or _log
    import uvicorn

    threading.Thread(target=_open, daemon=True).start()
    from app.main import app

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning", log_config=None)
