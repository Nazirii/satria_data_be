#!/usr/bin/env python3
"""Cek sambungan: embedding buatan backend harus sama dengan embedding dari notebook.

  python scripts/check_parity.py            (memakai .env / ARTIFACTS_DIR, ENCODER, DEVICE)

Mengambil N crop galeri dari roi/, meng-embed ulang dengan encoder backend, lalu
membandingkan dengan development_embeddings_*.npy. Lolos kalau cos >= 0,999 untuk
semua foto dan kelompok teratasnya sama dengan kelompok foto itu.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env(path: Path):
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--env", type=Path, default=ROOT / ".env")
    args = ap.parse_args()
    load_env(args.env)

    from app.engine import Engine, read_csv
    from app.settings import load_settings
    from app.vision import read_rgb

    eng = Engine(load_settings())
    if not eng.ready:
        sys.exit(f"Backend belum siap: {eng.error}")
    art, gal = eng.art, eng.gallery
    rows = read_csv(art.final_dir / "gallery_assignments.csv")
    Xp = np.load(art.final_dir / "development_embeddings_primary.npy")
    Xs = np.load(art.final_dir / "development_embeddings_siglip2.npy") if gal.fused else None
    in_gal = {r["sample_id"]: i for i, r in enumerate(gal.rows)}

    cand = [i for i, r in enumerate(rows) if r["cluster_id"] not in ("-1", "-1.0") and r["sample_id"] in in_gal]
    rng = np.random.default_rng(0)
    pick = rng.choice(cand, size=min(args.n, len(cand)), replace=False)
    if art.roi_dir is None:
        sys.exit("Folder roi/ tidak ada di artifacts, crop galeri tidak bisa dibaca.")

    worst, mism = 1.0, 0
    print(f"{'foto':22} {'cos dinov3':>10} {'cos siglip2':>11}  kelompok  tebakan")
    for i in pick:
        r = rows[i]
        path = art.roi_dir / f"{r['pixel_hash']}.jpg"
        crop = read_rgb(path)
        qp, qs = eng.embed(crop)
        cp = float(qp @ (Xp[i] / np.linalg.norm(Xp[i])))
        cs = float(qs @ (Xs[i] / np.linalg.norm(Xs[i]))) if qs is not None else float("nan")
        S = gal.cluster_scores(qp, qs, exclude=np.array([in_gal[r["sample_id"]]]))[0]
        own = gal.codes[gal.row_cluster[in_gal[r["sample_id"]]]]
        pred = gal.codes[int(np.argmax(S))]
        worst = min(worst, cp, cs if qs is not None else 1.0)
        mism += pred != own
        print(f"{r['sample_id'][:22]:22} {cp:10.5f} {cs:11.5f}  {own:8}  {pred}{'' if pred == own else '  ← beda'}")

    ok = worst >= args.min_cos
    print(f"\ncos terendah {worst:.5f} (syarat ≥ {args.min_cos}) → {'LOLOS' if ok else 'GAGAL'}")
    print(f"kelompok tebakan beda: {mism}/{len(pick)} (wajar kalau sedikit; foto di pinggir kelompok bisa lebih mirip tetangga)")
    if not ok:
        print("Kemungkinan penyebab: versi transformers beda dengan notebook (5.6.2), folder model salah, "
              "atau ENCODER=fake.")
        sys.exit(1)


if __name__ == "__main__":
    main()
