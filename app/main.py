from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))  # twilog_analyticsパッケージを参照可能にする

from app.routes import views  # sys.path調整後にインポートする

# FastAPIアプリ本体。HTMXを用いたSSR構成。
app = FastAPI(title="Twilog Analytics")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.include_router(views.router)
