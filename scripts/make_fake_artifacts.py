#!/usr/bin/env python3
"""Bikin artefak tiruan dengan format SAMA PERSIS dengan zip SuperGallery temanmu.

Dipakai untuk develop dan tes sebelum model asli ada. Embedding dibuat dengan
FakeEncoder (tanpa torch), jadi jalankan backend dengan ENCODER=fake.

  python scripts/make_fake_artifacts.py --images ../sipilah-fe/public/images/sample --out artifacts_fake

Struktur hasil (meniru SuperGallery_All_Artifacts.zip):
  <out>/supergallery/models/{dinov3,siglip2}/            (kosong; model tiruan tidak butuh bobot)
  <out>/supergallery/<kode>/retrieval_export/{config.json, retrieval_index.csv, index_primary.npy, index_siglip2.npy}
  <out>/supergallery/<kode>/supergallery_final/{gallery_assignments.csv, cluster_quality.csv,
                         development_embeddings_primary.npy, development_embeddings_siglip2.npy, manifest.json}
  <out>/supergallery/<kode>/roi/<pixel_hash>.jpg
  <out>/supergallery/<kode>/data_split.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.vision import FakeEncoder, read_rgb, roi_crop  # noqa: E402

KEY = "fake0000000000000000"


def variants(im: Image.Image, n: int, rng: np.random.Generator):
    """Beberapa versi foto yang sama: geser, potong, terang-gelap, cermin."""
    out = [im]
    w, h = im.size
    for _ in range(n - 1):
        v = im
        if rng.random() < 0.5:
            v = ImageOps.mirror(v)
        dx, dy = int(w * rng.uniform(0, 0.08)), int(h * rng.uniform(0, 0.08))
        v = v.crop((dx, dy, w - int(w * rng.uniform(0, 0.08)), h - int(h * rng.uniform(0, 0.08))))
        v = ImageEnhance.Brightness(v).enhance(rng.uniform(0.85, 1.15))
        out.append(v)
    return out


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True, help="folder foto, nama berawalan kXX- atau noise-")
    ap.add_argument("--out", type=Path, default=Path("artifacts_fake"))
    ap.add_argument("--views", type=int, default=6, help="jumlah versi per foto (1 dipakai sebagai foto uji)")
    ap.add_argument("--dim", type=int, default=768)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    base = args.out / "supergallery"
    run = base / KEY
    for d in ("retrieval_export", "supergallery_final", "roi"):
        (run / d).mkdir(parents=True, exist_ok=True)
    for m in ("dinov3", "siglip2"):
        (base / "models" / m).mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    prefixes = sorted({p.stem.split("-")[0] for p in files if not p.stem.startswith("noise")})
    cid_of = {pre: i for i, pre in enumerate(prefixes)}  # id kelompok rapat 0..C-1, seperti HDBSCAN

    enc_p = FakeEncoder("dinov3", args.dim, seed=0)
    enc_s = FakeEncoder("siglip2", args.dim, seed=1)

    study, split_rows, crops = [], [], []
    for p in files:
        pre = p.stem.split("-")[0]
        cid = -1 if pre == "noise" else cid_of[pre]
        for j, v in enumerate(variants(read_rgb(p), args.views, rng)):
            ph = hashlib.sha256(v.tobytes()).hexdigest()
            crop, _ = roi_crop(v)
            crop_path = run / "roi" / f"{ph}.jpg"
            crop.save(crop_path, quality=95)
            split = "test" if j == args.views - 1 else ("gallery" if j % 2 == 0 else "val")
            sid = ph[:20]
            row = {"sample_id": sid, "image_path": f"/kaggle/working/fake/{p.stem}_{j}.jpg",
                   "crop_path": f"/kaggle/working/supergallery/{KEY}/roi/{ph}.jpg", "split": split,
                   "product_id": "", "pixel_hash": ph}
            split_rows.append({**row, "cluster_hint": cid})
            if split != "test":
                study.append({**row, "cluster_id": cid})
                crops.append(read_rgb(crop_path))  # baca ulang dari JPEG, persis seperti notebook

    Xp = enc_p.embed(crops)
    Xs = enc_s.embed(crops)
    labels = np.array([r["cluster_id"] for r in study])

    # kualitas per kelompok (versi sederhana dari sel 05 notebook)
    clusters = sorted(int(c) for c in set(labels.tolist()) - {-1})
    cents = {c: Xp[labels == c].mean(0) / np.linalg.norm(Xp[labels == c].mean(0)) for c in clusters}
    C = np.stack([cents[c] for c in clusters])
    quality = []
    for i, c in enumerate(clusters):
        idx = np.flatnonzero(labels == c)
        comp = float((Xp[idx] @ cents[c]).mean())
        cc = C @ C.T
        np.fill_diagonal(cc, -np.inf)
        near = float(cc[i].max())
        sil = float(np.clip(comp - near, -1, 1))
        ok = len(idx) >= 3 and comp >= 0.5 and (comp - near) >= 0.02
        quality.append({"cluster_id": c, "n_samples": len(idx), "compactness": round(comp, 6), "medoid_row": int(idx[0]),
                        "nearest_cluster_similarity": round(near, 6), "cluster_margin": round(comp - near, 6),
                        "mean_silhouette": round(sil, 6), "method_stability_ari": 0.8, "quality_pass": ok,
                        "quality_status": "PASS" if ok else "REVIEW", "medoid_sample_id": study[idx[0]]["sample_id"]})
    status = {q["cluster_id"]: q["quality_status"] for q in quality}
    assign = []
    for r in study:
        st = "NOISE" if r["cluster_id"] == -1 else status[r["cluster_id"]]
        assign.append({**r, "quality_status": st, "quality_pass": st == "PASS"})

    fin = run / "supergallery_final"
    write_csv(fin / "gallery_assignments.csv", assign)
    write_csv(fin / "cluster_quality.csv", quality)
    np.save(fin / "development_embeddings_primary.npy", Xp)
    np.save(fin / "development_embeddings_siglip2.npy", Xs)
    gate = {"min_support": 3, "min_compactness": 0.5, "min_silhouette": 0.0, "min_margin": 0.02, "min_stability": 0.46}
    manifest = {
        "stage": "05_frozen_supergallery", "data_key": KEY, "cluster_method": "HDBSCAN",
        "encoder": {"backbone": "dinov3", "pooling": "cls", "resolution": 224, "l2_normalized": True,
                    "roi_version": "canny_bbox_v1"},
        "cluster_gate": gate, "pass_clusters": [q["cluster_id"] for q in quality if q["quality_pass"]],
        "routing_clusters": clusters, "quality_fallback": False, "product_id_available": False,
        "retrieval_representation": {"primary": "dinov3", "use_siglip2_fusion": True, "primary_weight": 0.55,
                                     "siglip2_pooling": "native", "siglip2_resolution": 224},
        "contract": "FAKE artifacts for development only.",
    }
    (fin / "manifest.json").write_text(json.dumps(manifest, indent=2))

    gal = [i for i, r in enumerate(assign) if r["split"] == "gallery" and r["cluster_id"] != -1]
    ex = run / "retrieval_export"
    write_csv(ex / "retrieval_index.csv", [{**assign[i], "emb_row": i, "identity_id": f"cluster_{assign[i]['cluster_id']}"}
                                          for i in gal])
    np.save(ex / "index_primary.npy", Xp[gal])
    np.save(ex / "index_siglip2.npy", Xs[gal])
    config = {
        "retrieval": {"top_k": 5, "route_top_clusters": 2, "min_candidates": 8, "default_mode": "cluster_guided",
                      "primary_weight": 0.55, "use_siglip2": True},
        "recognition_gate": {"tau_a": 0.88, "tau_u": 0.6, "margin": 0.04, "source": "fallback_no_product_identity_gt"},
        "admission_gate": {"tau_update": 0.9, "margin_update": 0.05, "source": "provisional_fallback"},
        "backbone": "dinov3",
        "supergallery_manifest": manifest,
    }
    (ex / "config.json").write_text(json.dumps(config, indent=2))
    write_csv(run / "data_split.csv", split_rows)
    print(f"{len(assign)} foto berklaster/noise, {len(clusters)} kelompok, {len(gal)} foto index, "
          f"{sum(r['split'] == 'test' for r in split_rows)} foto uji → {args.out}")


if __name__ == "__main__":
    main()
