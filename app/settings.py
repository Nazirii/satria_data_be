"""Setelan backend. Semua dibaca dari environment (uvicorn --env-file .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _float(name: str) -> float | None:
    v = os.getenv(name, "").strip()
    return float(v) if v else None


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default)).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path
    encoder: str            # "model" | "fake"
    device: str             # "auto" | "cpu" | "cuda"
    gallery: str            # "all" (galeri + validasi) | "index" (persis retrieval_export)
    tau_accept: float | None
    tau_margin: float | None
    dedup_cos: float
    dedup_sec: float
    fe_dist: Path
    data_dir: Path
    max_upload_mb: float
    threads: int
    cors_origins: tuple[str, ...]


def load_settings() -> Settings:
    return Settings(
        artifacts_dir=_path("ARTIFACTS_DIR", "artifacts"),
        encoder=os.getenv("ENCODER", "model").strip().lower(),
        device=os.getenv("DEVICE", "auto").strip().lower(),
        gallery=os.getenv("GALLERY", "all").strip().lower(),
        tau_accept=_float("TAU_ACCEPT"),
        tau_margin=_float("TAU_MARGIN"),
        dedup_cos=_float("DEDUP_COS") or 0.95,
        dedup_sec=_float("DEDUP_SEC") or 3.0,
        fe_dist=_path("FE_DIST", "../sipilah-fe/dist"),
        data_dir=_path("DATA_DIR", "data"),
        max_upload_mb=_float("MAX_UPLOAD_MB") or 15.0,
        threads=int(os.getenv("TORCH_THREADS", "0") or 0),
        cors_origins=tuple(
            o.strip() for o in os.getenv(
                "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",") if o.strip()
        ),
    )
