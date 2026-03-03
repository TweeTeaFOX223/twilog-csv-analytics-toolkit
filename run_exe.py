from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn


def _ensure_src_path() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))


def main() -> None:
    _ensure_src_path()
    from app.main import app as fastapi_app

    host = os.getenv("TWILOG_HOST", "127.0.0.1")
    port = int(os.getenv("TWILOG_PORT", "8000"))
    open_browser = os.getenv("TWILOG_OPEN_BROWSER", "1").lower() not in {"0", "false", "no"}
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{browser_host}:{port}"

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    config = uvicorn.Config(fastapi_app, host=host, port=port, reload=False, log_level="info")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
