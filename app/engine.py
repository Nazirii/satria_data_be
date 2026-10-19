"""Mesin pengenal: memuat hasil notebook SuperGallery, memberi skor, dan memutuskan.

Alur per foto (sama dengan `retrieve_evidence(mode="global")` + `recognition_decision`
di notebook teman):
  skor ke tiap foto galeri = w * cos(DINOv3) + (1 - w) * cos(SigLIP2)
  skor kelompok            = skor tertinggi di antara foto anggotanya
  dikenali                 = skor #1 >= tau_accept  dan  (skor #1 - skor #2) >= tau_margin
"""
from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .settings import Settings
from .vision import FakeEncoder, HFEncoder, bbox_rel, l2, roi_crop

NOISE = -1


class ArtifactError(RuntimeError):
    """Isi folder artifacts tidak lengkap atau tidak cocok."""


# --------------------------------------------------------------------------- artefak

@dataclass
class Artifacts:
    root: Path
    run_dir: Path               # .../supergallery/<kode data>
    config: dict
    export_dir: Path            # retrieval_export/
    final_dir: Path             # supergallery_final/
    models_dir: Path | None     # models/
    roi_dir: Path | None        # roi/ (crop untuk thumbnail & cek)

    @property
    def manifest(self) -> dict:
        return self.config.get("supergallery_manifest", {})

    @property
    def encoder_cfg(self) -> dict:
        return self.manifest.get("encoder", {})

    @property
    def retrieval_rep(self) -> dict:
        return self.manifest.get("retrieval_representation", {})

    @property
    def use_siglip(self) -> bool:
        return bool(self.config.get("retrieval", {}).get("use_siglip2", False))

    @property
    def primary_weight(self) -> float:
        return float(self.config.get("retrieval", {}).get("primary_weight", 1.0))


def find_artifacts(root: Path) -> Artifacts:
    """Cari isi zip SuperGallery di bawah `root`, sedalam apa pun foldernya."""
    root = Path(root)
    if not root.exists():
        raise ArtifactError(f"Folder artifacts tidak ada: {root}")
    configs = sorted(root.rglob("retrieval_export/config.json"))
    if not configs:
        raise ArtifactError(f"retrieval_export/config.json tidak ditemukan di {root}")
    if len(configs) > 1:
        configs.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # pakai run terbaru
    export_dir = configs[0].parent
    run_dir = export_dir.parent
    final_dir = run_dir / "supergallery_final"
    if not final_dir.exists():
        raise ArtifactError(f"supergallery_final/ tidak ditemukan di {run_dir}")
    models = [p for p in (run_dir.parent / "models", run_dir / "models") if p.exists()]
    if not models:
        models = sorted(p for p in root.rglob("models") if p.is_dir())
    roi = run_dir / "roi"
    return Artifacts(
        root=root,
        run_dir=run_dir,
        config=json.loads(configs[0].read_text()),
        export_dir=export_dir,
        final_dir=final_dir,
        models_dir=models[0] if models else None,
        roi_dir=roi if roi.exists() else None,
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cluster_code(cid: int, width: int = 2) -> str:
    return "noise" if cid == NOISE else f"K{cid:0{width}d}"


# --------------------------------------------------------------------------- galeri

@dataclass
class Gallery:
    """Foto referensi yang sudah punya kelompok, disusun urut per kelompok."""

    rows: list[dict]            # metadata per foto (urutan sama dengan matriks di bawah)
    primary: np.ndarray         # N x D, L2
    secondary: np.ndarray | None
    weight: float
    cluster_ids: np.ndarray     # (C,) id kelompok asli (int)
    codes: list[str]            # (C,) "K00", ...
    starts: np.ndarray          # awal tiap kelompok di matriks (untuk reduceat)
    row_cluster: np.ndarray     # (N,) indeks kelompok 0..C-1 per foto
    sizes: np.ndarray = field(default=None)

    @classmethod
    def build(cls, rows, primary, secondary, weight, code_width=None):
        cid = np.array([int(float(r["cluster_id"])) for r in rows])
        keep = cid != NOISE
        rows = [r for r, k in zip(rows, keep) if k]
        cid = cid[keep]
        primary = l2(np.asarray(primary, np.float32)[keep])
        secondary = None if secondary is None else l2(np.asarray(secondary, np.float32)[keep])
        if len(rows) == 0:
            raise ArtifactError("Galeri kosong: tidak ada foto yang punya kelompok")
        order = np.argsort(cid, kind="stable")
        rows = [rows[i] for i in order]
        cid, primary = cid[order], primary[order]
        secondary = None if secondary is None else secondary[order]
        ids, starts, sizes = np.unique(cid, return_index=True, return_counts=True)
        width = code_width or max(2, len(str(int(ids.max()))))
        row_cluster = np.searchsorted(ids, cid)
        return cls(rows, primary, secondary, float(weight), ids, [cluster_code(int(i), width) for i in ids],
                   starts, row_cluster, sizes)

    @property
    def fused(self) -> bool:
        return self.secondary is not None

    def similarity(self, qp: np.ndarray, qs: np.ndarray | None = None) -> np.ndarray:
        """(B, N) skor gabungan ke setiap foto galeri."""
        qp = np.atleast_2d(qp)
        s = qp @ self.primary.T
        if self.fused and qs is not None:
            s = self.weight * s + (1.0 - self.weight) * (np.atleast_2d(qs) @ self.secondary.T)
        return s

    def cluster_scores(self, qp, qs=None, exclude: np.ndarray | None = None) -> np.ndarray:
        """(B, C) skor tertinggi per kelompok. `exclude[b]` = baris galeri yang diabaikan (-1 = tidak ada)."""
        s = self.similarity(qp, qs)
        if exclude is not None:
            b = np.flatnonzero(np.asarray(exclude) >= 0)
            s[b, np.asarray(exclude)[b]] = -np.inf
        return np.maximum.reduceat(s, self.starts, axis=1)


def load_gallery(art: Artifacts, mode: str = "all") -> Gallery:
    """mode "all": semua foto berklaster (galeri + validasi). mode "index": persis retrieval_export."""
    if mode == "index":
        rows = read_csv(art.export_dir / "retrieval_index.csv")
        primary = np.load(art.export_dir / "index_primary.npy")
        sec_path = art.export_dir / "index_siglip2.npy"
    else:
        rows = read_csv(art.final_dir / "gallery_assignments.csv")
        primary = np.load(art.final_dir / "development_embeddings_primary.npy")
        sec_path = art.final_dir / "development_embeddings_siglip2.npy"
    secondary = np.load(sec_path) if art.use_siglip and sec_path.exists() else None
    if art.use_siglip and secondary is None:
        raise ArtifactError(f"Config memakai SigLIP2 tapi {sec_path.name} tidak ada")
    if len(rows) != len(primary) or (secondary is not None and len(secondary) != len(rows)):
        raise ArtifactError(
            f"Jumlah baris tidak cocok: CSV {len(rows)}, primary {len(primary)}, "
            f"siglip2 {None if secondary is None else len(secondary)}")
    # lebar kode mengikuti seluruh klaster (bukan hanya yang ada di galeri)
    all_ids = [int(float(r["cluster_id"])) for r in read_csv(art.final_dir / "gallery_assignments.csv")]
    width = max(2, len(str(max(all_ids))))
    return Gallery.build(rows, primary, secondary, art.primary_weight if secondary is not None else 1.0, width)


# --------------------------------------------------------------------------- keputusan

@dataclass(frozen=True)
class Thresholds:
    accept: float
    margin: float
    unknown: float
    source: str


def decide(top1: float, top2: float, t: Thresholds) -> tuple[str, str | None, str]:
    """→ (status FE, alasan, status asli notebook)."""
    margin = top1 - top2
    if top1 < t.unknown:
        gate = "UNKNOWN"
    elif top1 >= t.accept and margin >= t.margin:
        gate = "KNOWN"
    else:
        gate = "REVIEW"
    if gate == "KNOWN":
        return "dikenali", None, gate
    return "belum", ("low_score" if top1 < t.accept else "ambiguous"), gate


# --------------------------------------------------------------------------- mesin

class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.error: str | None = None
        self.lock = threading.Lock()
        self._last: tuple[np.ndarray, float, dict] | None = None
        try:
            self._load()
            self.ready = True
        except Exception as e:  # server tetap hidup; FE pindah ke simulasi
            self.error = f"{type(e).__name__}: {e}"

    # ---- muat
    def _load(self):
        s = self.settings
        self.art = find_artifacts(s.artifacts_dir)
        enc = self.art.encoder_cfg
        if enc.get("roi_version", "canny_bbox_v1") != "canny_bbox_v1":
            raise ArtifactError(f"Versi potong {enc.get('roi_version')} belum didukung backend")
        self.gallery = load_gallery(self.art, s.gallery)
        rep = self.art.retrieval_rep
        backbone = enc.get("backbone") or self.art.config.get("backbone", "dinov3")
        res_p = int(enc.get("resolution", 224))
        res_s = int(rep.get("siglip2_resolution", 224))

        if s.encoder == "fake":
            self.primary = FakeEncoder(backbone, self.gallery.primary.shape[1], res_p, seed=0)
            self.secondary = FakeEncoder("siglip2", self.gallery.secondary.shape[1], res_s, seed=1) \
                if self.gallery.fused else None
        else:
            if self.art.models_dir is None:
                raise ArtifactError("Folder models/ tidak ditemukan di artifacts")
            self.primary = HFEncoder(backbone, self.art.models_dir / backbone, enc.get("pooling", "cls"),
                                     res_p, s.device, s.threads)
            self.secondary = HFEncoder("siglip2", self.art.models_dir / "siglip2",
                                       rep.get("siglip2_pooling", "native"), res_s, s.device, s.threads) \
                if self.gallery.fused else None

        gate = self.art.config.get("recognition_gate", {})
        from_env = s.tau_accept is not None or s.tau_margin is not None
        accept = s.tau_accept if s.tau_accept is not None else float(gate.get("tau_a", 0.88))
        self.thresholds = Thresholds(
            accept=accept,
            margin=s.tau_margin if s.tau_margin is not None else float(gate.get("margin", 0.04)),
            unknown=min(float(gate.get("tau_u", 0.60)), accept),
            source="env" if from_env else f"config ({gate.get('source', '?')})",
        )
        self.embed(Image.new("RGB", (224, 224), (200, 200, 200)))  # pemanasan

    # ---- inti
    def embed(self, crop: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
        qp = self.primary.embed([crop])[0]
        qs = self.secondary.embed([crop])[0] if self.secondary is not None else None
        return qp, qs

    def rank(self, qp, qs, k: int = 5):
        scores = self.gallery.cluster_scores(qp, qs)[0]
        order = np.argsort(-scores, kind="stable")
        top1 = float(scores[order[0]])
        top2 = float(scores[order[1]]) if len(order) > 1 else 0.0
        top = [{"cluster": self.gallery.codes[j], "sim": round(float(scores[j]), 4)} for j in order[:k]]
        return top, top1, top2

    def infer(self, image: Image.Image, source: str = "photo") -> dict:
        t0 = time.perf_counter()
        with self.lock:
            crop, box = roi_crop(image)
            qp, qs = self.embed(crop)
            top, top1, top2 = self.rank(qp, qs)
            status, reason, gate = decide(top1, top2, self.thresholds)
            obj = {
                "bbox": bbox_rel(box, image.size),
                "status": status,
                "cluster": top[0]["cluster"] if status == "dikenali" else None,
                "score": round(top1, 4),
                "margin": round(top1 - top2, 4),
                "reason": reason,
                "gate": gate,
                "top": top,
            }
            duplicate = False
            now = time.monotonic()
            if source == "live" and self._last is not None:
                last_q, last_t, last_obj = self._last
                if now - last_t <= self.settings.dedup_sec and float(qp @ last_q) >= self.settings.dedup_cos:
                    duplicate, obj = True, last_obj
            if source == "live" and not duplicate:
                self._last = (qp, now, obj)
        return {"objects": [obj], "latency_ms": round((time.perf_counter() - t0) * 1000), "duplicate": duplicate}

    # ---- info
    def health(self) -> dict:
        if not self.ready:
            return {"ok": False, "error": self.error, "mode": self.settings.encoder}
        g = self.gallery
        return {
            "ok": True,
            "mode": self.settings.encoder,
            "device": getattr(self.primary, "device", "cpu"),
            "encoders": [self.primary.name] + ([self.secondary.name] if self.secondary else []),
            "fusion_weight": g.weight if g.fused else 1.0,
            "gallery": {"source": self.settings.gallery, "images": len(g.rows), "clusters": len(g.codes)},
            "thresholds": {"accept": self.thresholds.accept, "margin": self.thresholds.margin,
                           "source": self.thresholds.source},
            "artifacts": str(self.art.run_dir),
        }
