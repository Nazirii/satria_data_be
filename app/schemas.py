"""Bentuk JSON yang dipertukarkan dengan frontend (lihat sipilah-fe/src/services/api.js)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TopItem(BaseModel):
    cluster: str
    sim: float


class DetectedObject(BaseModel):
    bbox: list[float] = Field(description="[x, y, w, h] relatif 0–1 terhadap gambar yang dikirim")
    status: Literal["dikenali", "belum"]
    cluster: Optional[str] = Field(description="kode kelompok bila dikenali, mis. K06")
    score: float = Field(description="skor kelompok teratas (0–1)")
    margin: float = Field(description="skor #1 dikurangi skor #2")
    reason: Optional[Literal["low_score", "ambiguous"]] = Field(
        description="alasan 'belum': kurang mirip, atau mirip dua kelompok sekaligus")
    gate: Literal["KNOWN", "REVIEW", "UNKNOWN"]
    top: list[TopItem]


class InferResponse(BaseModel):
    objects: list[DetectedObject]
    latency_ms: int
    duplicate: bool = False


class Feedback(BaseModel):
    item_id: str
    cluster: Optional[str] = None       # kelompok yang ditebak sistem
    flagged: bool = True                # True = ditandai salah kelompok, False = tanda dibatalkan
    score: Optional[float] = None
    source: Optional[str] = None        # live | photo | gallery
    note: Optional[str] = None
    image: Optional[str] = Field(default=None, description="thumbnail data URL (opsional)")
