"""Bagian visual: buka foto, potong area barang, kanvas 224, dan encoder.

Fungsi `roi_crop` dan `letterbox` disalin dari notebook SuperGallery (sel 02 dan 03B)
dengan konstanta yang sama. Jangan diubah tanpa membuat ulang embedding galeri,
karena galeri dibangun dari potongan yang dihasilkan fungsi persis ini.
"""
from __future__ import annotations

import inspect
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # foto HEIC dari iPhone
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover - opsional
    pass

ROI_MARGIN = 0.10
ROI_DETECT_MAX = 640
PAD_COLOR = (128, 128, 128)


class BadImage(ValueError):
    """Berkas yang dikirim bukan gambar yang bisa dibaca."""


def read_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return ImageOps.exif_transpose(im).convert("RGB").copy()
    except Exception as e:
        raise BadImage(f"Berkas bukan gambar yang bisa dibaca ({type(e).__name__})") from e


def read_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB").copy()


def roi_crop(im: Image.Image):
    """Salinan `roi_crop` notebook: kontur Canny terbesar + margin 10%.

    Balikan: (crop, (x1, y1, x2, y2)) dalam piksel gambar asli.
    """
    img = np.asarray(im)
    h, w = img.shape[:2]

    scale = max(h, w) / ROI_DETECT_MAX
    if scale > 1:
        small = cv2.resize(img, (round(w / scale), round(h / scale)), interpolation=cv2.INTER_AREA)
    else:
        small, scale = img, 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 50, 150)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    edge = cv2.dilate(edge, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return im, (0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    mx, my = int(bw * ROI_MARGIN), int(bh * ROI_MARGIN)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(small.shape[1], x + bw + mx), min(small.shape[0], y + bh + my)
    bbox = (round(x1 * scale), round(y1 * scale), min(w, round(x2 * scale)), min(h, round(y2 * scale)))
    return im.crop(bbox), bbox


def bbox_rel(bbox, size) -> list[float]:
    """(x1, y1, x2, y2) piksel → [x, y, w, h] relatif 0–1, seperti yang dipakai FE."""
    x1, y1, x2, y2 = bbox
    w, h = size
    return [round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)]


def letterbox(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_array(im: Image.Image, size: int, mean, std) -> np.ndarray:
    """Kanvas → array CHW float32 ternormalisasi (setara TF.pil_to_tensor/255 + TF.normalize)."""
    x = np.asarray(letterbox(im, size), dtype=np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- encoder

class FakeEncoder:
    """Encoder tiruan tanpa torch, untuk develop dan tes.

    Foto yang mirip secara piksel menghasilkan vektor yang mirip, jadi seluruh alur
    (potong → embed → skor → keputusan) bisa dicoba tanpa model asli.
    """

    def __init__(self, name: str, dim: int = 768, resolution: int = 224, seed: int = 0):
        self.name, self.dim, self.resolution = name, dim, resolution
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((16 * 16 * 3, dim)).astype(np.float32)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        rows = []
        for im in images:
            x = np.asarray(letterbox(im, 64).resize((16, 16), Image.Resampling.BILINEAR), np.float32) / 255.0
            x = (x - x.mean()).reshape(-1)
            rows.append(x @ self.proj)
        return l2(np.stack(rows)).astype(np.float32)


class HFEncoder:
    """Encoder Hugging Face dari folder `save_pretrained` (models/dinov3, models/siglip2)."""

    def __init__(self, name: str, model_dir: Path, pooling: str, resolution: int, device: str = "auto", threads: int = 0):
        import torch

        if threads > 0:
            torch.set_num_threads(threads)
        self.torch = torch
        self.name, self.pooling, self.resolution = name, pooling, resolution
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        cfg = json.loads((model_dir / "config.json").read_text())
        pre = model_dir / "preprocessor_config.json"
        if pre.exists():
            p = json.loads(pre.read_text())
            self.mean, self.std = p.get("image_mean"), p.get("image_std")
        else:
            self.mean = self.std = None
        if not self.mean or not self.std:  # sama dengan fallback notebook
            self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        model_type = str(cfg.get("model_type", "")).lower()
        if "siglip" in model_type:
            from transformers import SiglipVisionModel as cls
        else:
            from transformers import AutoModel as cls
        model = cls.from_pretrained(model_dir, dtype=torch.float32)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.kwargs = (
            {"interpolate_pos_encoding": True}
            if "interpolate_pos_encoding" in inspect.signature(self.model.forward).parameters
            else {}
        )
        self.n_reg = getattr(self.model.config, "num_register_tokens", 0)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        x = np.stack([to_array(im, self.resolution, self.mean, self.std) for im in images])
        with torch.inference_mode():
            out = self.model(pixel_values=torch.from_numpy(x).to(self.device), **self.kwargs)
            tokens = out.last_hidden_state
            if self.pooling == "native":
                pooled = getattr(out, "pooler_output", None)
                z = tokens[:, 0] if pooled is None else pooled
            elif self.pooling == "cls":
                z = tokens[:, 0]
            elif self.pooling == "mean_patch":
                z = tokens[:, 1 + self.n_reg:].mean(dim=1)
            else:
                raise ValueError(f"Pooling '{self.pooling}' belum didukung backend")
            z = torch.nn.functional.normalize(z.float(), dim=1)
        return z.cpu().numpy().astype(np.float32)
