"""SiPilah backend (FastAPI). Jalankan:  uvicorn app.main:app --env-file .env"""
from __future__ import annotations

import base64
import json
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .engine import Engine
from .schemas import Feedback, InferResponse
from .settings import load_settings
from .vision import BadImage, read_image

settings = load_settings()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = Engine(settings)
    eng = state["engine"]
    print("SiPilah:", "siap" if eng.ready else f"BELUM SIAP → {eng.error}", flush=True)
    yield


app = FastAPI(title="SiPilah API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"],
                   allow_headers=["*"])


def engine() -> Engine:
    eng = state.get("engine")
    if eng is None or not eng.ready:
        raise HTTPException(503, detail=f"Model belum siap: {eng.error if eng else 'belum dimuat'}")
    return eng


@app.get("/health")
def health():
    eng = state.get("engine")
    return eng.health() if eng else {"ok": False, "error": "belum dimuat"}


@app.post("/infer", response_model=InferResponse)
def infer(image: UploadFile = File(...), source: str = Query("photo", pattern="^(photo|live)$")):
    eng = engine()
    data = image.file.read(int(settings.max_upload_mb * 1024 * 1024) + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, detail=f"Foto lebih dari {settings.max_upload_mb:g} MB")
    try:
        im = read_image(data)
    except BadImage as e:
        raise HTTPException(400, detail=str(e))
    return eng.infer(im, source=source)


DATA_URL = re.compile(r"^data:image/(jpeg|png|webp);base64,(.+)$", re.S)


@app.post("/feedback")
def feedback(fb: Feedback):
    out = settings.data_dir / "feedback"
    out.mkdir(parents=True, exist_ok=True)
    rec = fb.model_dump(exclude={"image"})
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if fb.image and (m := DATA_URL.match(fb.image)):
        raw = base64.b64decode(m.group(2))
        if len(raw) <= 2 * 1024 * 1024:
            name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.{m.group(1).replace('jpeg', 'jpg')}"
            (out / name).write_bytes(raw)
            rec["image_file"] = name
    with (settings.data_dir / "feedback.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


# Hasil build FE disajikan di "/" supaya cukup satu alamat (didaftarkan paling akhir).
if (settings.fe_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=settings.fe_dist, html=True), name="fe")
