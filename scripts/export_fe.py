#!/usr/bin/env python3
"""Isi data frontend dari hasil notebook SuperGallery. Tidak butuh model/torch.

  python scripts/export_fe.py --artifacts artifacts --fe ../sipilah-fe

Menulis ke FE:
  public/data/clusters.json            ringkasan kelompok (nama/wadah/bahaya dari cluster_labels.csv)
  public/data/clusters/<ID>.json       foto anggota: skor, urutan, 5 kelompok paling mirip, 15% terjauh
  public/images/clusters/<ID>/*.jpg    thumbnail dari crop roi/
Menulis ke backend:
  cluster_labels.csv                   (dibuat kalau belum ada) isi nama, keluarga, wadah, bahaya, langkah
  review/contact_sheets/<ID>.jpg       satu lembar contoh foto per kelompok, untuk memberi nama

Skor dihitung dengan fungsi yang sama dengan backend (skor tertinggi per kelompok,
gabungan DINOv3 + SigLIP2), dengan foto itu sendiri dikecualikan.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.engine import NOISE, cluster_code, find_artifacts, load_gallery, read_csv  # noqa: E402

LABEL_COLS = ["cluster", "foto", "name", "family", "bin", "hazard", "steps", "visual"]
FAMILIES = "listrik | audio | energi | rumah | komputer | tampilan | campuran"
BINS = "b3 | layar | gawai | ti | kecil | besar | pcb | karantina"


def crop_file(art, row) -> Path | None:
    if art.roi_dir is None:
        return None
    p = art.roi_dir / f"{row['pixel_hash']}.jpg"
    if p.exists():
        return p
    p = art.roi_dir / Path(row.get("crop_path", "")).name
    return p if p.exists() else None


def thumb(src: Path, dst: Path, size: int) -> bool:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=84, optimize=True)
        return True
    except Exception as e:
        print(f"  lewati {src.name}: {e}", file=sys.stderr)
        return False


def contact_sheet(paths: list[Path], title: str, dst: Path, cols=6, cell=170):
    paths = paths[:24]
    rows = max(1, (len(paths) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell + 40), (245, 246, 244))
    d = ImageDraw.Draw(sheet)
    d.text((10, 12), title, fill=(20, 24, 22))
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((cell - 8, cell - 8))
            x, y = (i % cols) * cell + (cell - im.width) // 2, 40 + (i // cols) * cell + (cell - im.height) // 2
            sheet.paste(im, (x, y))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, "JPEG", quality=85)


def load_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["cluster"].strip(): r for r in csv.DictReader(f) if r.get("cluster")}


def write_label_template(path: Path, codes: list[str], sizes: dict, old: dict):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLS)
        w.writeheader()
        for c in codes:
            r = old.get(c, {})
            w.writerow({"cluster": c, "foto": sizes.get(c, 0), "name": r.get("name", ""),
                        "family": r.get("family", ""), "bin": r.get("bin", ""), "hazard": r.get("hazard", ""),
                        "steps": r.get("steps", ""), "visual": r.get("visual", "")})


def fnum(v):
    try:
        x = float(v)
        return None if np.isnan(x) else round(x, 4)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--fe", type=Path, default=ROOT.parent / "sipilah-fe")
    ap.add_argument("--labels", type=Path, default=ROOT / "cluster_labels.csv")
    ap.add_argument("--gallery", choices=["all", "index"], default="all", help="samakan dengan GALLERY di .env")
    ap.add_argument("--thumb", type=int, default=360)
    ap.add_argument("--edge-share", type=float, default=0.15)
    ap.add_argument("--sheets", type=Path, default=ROOT / "review" / "contact_sheets")
    args = ap.parse_args()

    art = find_artifacts(args.artifacts)
    gal = load_gallery(art, args.gallery)
    rows = read_csv(art.final_dir / "gallery_assignments.csv")
    Xp = np.load(art.final_dir / "development_embeddings_primary.npy").astype(np.float32)
    Xs = np.load(art.final_dir / "development_embeddings_siglip2.npy").astype(np.float32) if gal.fused else None
    Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
    if Xs is not None:
        Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    width = len(gal.codes[0]) - 1
    code_of = lambda r: cluster_code(int(float(r["cluster_id"])), width)  # noqa: E731
    in_gal = {r["sample_id"]: i for i, r in enumerate(gal.rows)}
    print(f"{len(rows)} foto, {len(gal.codes)} kelompok, galeri referensi {len(gal.rows)} foto ({args.gallery})")

    # 1. skor tiap foto (tanpa dirinya sendiri)
    tops, own = [], []
    for a in range(0, len(rows), 512):
        b = min(len(rows), a + 512)
        excl = np.array([in_gal.get(r["sample_id"], -1) for r in rows[a:b]])
        S = gal.cluster_scores(Xp[a:b], None if Xs is None else Xs[a:b], exclude=excl)
        for k, r in enumerate(rows[a:b]):
            order = np.argsort(-S[k])[:5]
            tops.append([{"cluster": gal.codes[j], "sim": round(float(S[k, j]), 4)} for j in order])
            c = code_of(r)
            own.append(round(float(S[k, gal.codes.index(c)]), 4) if c in gal.codes else round(float(S[k, order[0]]), 4))

    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(code_of(r), []).append(i)

    # 2. label kelompok
    sizes = {c: len(v) for c, v in groups.items() if c != "noise"}
    old = load_labels(args.labels)
    write_label_template(args.labels, gal.codes, sizes, old)
    labels = load_labels(args.labels)
    named = sum(bool(labels.get(c, {}).get("name", "").strip()) for c in gal.codes)
    print(f"cluster_labels.csv: {named}/{len(gal.codes)} kelompok sudah bernama → {args.labels}")

    # 3. tulis per kelompok
    pub = args.fe / "public"
    for d in (pub / "data" / "clusters", pub / "images" / "clusters"):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    quality = {cluster_code(int(float(q["cluster_id"])), width): q for q in read_csv(art.final_dir / "cluster_quality.csv")}
    summary_path = pub / "data" / "clusters.json"
    prev = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    gate_cfg = art.manifest.get("cluster_gate", {})
    t = {"support": int(gate_cfg.get("min_support", 5)), "silhouette": fnum(gate_cfg.get("min_silhouette")) or 0.0,
         "compactness": fnum(gate_cfg.get("min_compactness")) or 0.0, "margin": fnum(gate_cfg.get("min_margin")) or 0.0}

    clusters, noise = [], None
    for cid in sorted(groups):
        idx = groups[cid]
        if cid != "noise":
            idx = sorted(idx, key=lambda i: -own[i])
        edge_from = int(len(idx) * (1 - args.edge_share))
        items, sheet_src = [], []
        for n, i in enumerate(idx):
            r = rows[i]
            src = crop_file(art, r)
            rel = f"images/clusters/{cid}/{r['sample_id']}.jpg"
            ok = bool(src) and thumb(src, pub / rel, args.thumb)
            if ok:
                sheet_src.append(pub / rel)
            it = {"id": f"{cid}-{n + 1:04d}", "file": Path(r.get("image_path", "")).name or r["sample_id"],
                  "src": rel if ok else None, "sim": own[i], "rank": n + 1, "top": tops[i]}
            if cid == "noise":
                it["nearest"] = tops[i][0]["cluster"]
                it["margin"] = round(tops[i][0]["sim"] - (tops[i][1]["sim"] if len(tops[i]) > 1 else 0), 4)
            else:
                it["edge"] = n >= edge_from
            items.append(it)
        (pub / "data" / "clusters" / f"{cid}.json").write_text(
            json.dumps({"id": cid, "items": items}, ensure_ascii=False), encoding="utf-8")
        samples = [{"src": it["src"], "sim": it["sim"]} for it in items if it["src"]][:6]
        lab = labels.get(cid, {})
        if sheet_src:
            title = f"{cid} · {len(idx)} foto · {lab.get('name') or 'belum bernama'}"
            contact_sheet(sheet_src, title, args.sheets / f"{cid}.jpg")

        if cid == "noise":
            noise = {"id": "noise", "name": "Belum terkelompok", "family": "noise", "status": "karantina",
                     "size": len(idx), "bin": "karantina", "hazard": "sedang", "visual": "", "components": [],
                     "hazardNote": "", "steps": [], "note": "", "sources": [], "samples": samples}
            continue
        q = quality.get(cid, {})
        sil, comp, mar = fnum(q.get("mean_silhouette")), fnum(q.get("compactness")), fnum(q.get("cluster_margin"))
        clusters.append({
            "id": cid,
            "name": lab.get("name", "").strip() or f"Kelompok {cid}",
            "family": lab.get("family", "").strip() or "campuran",
            "status": "lolos" if q.get("quality_status", "PASS") == "PASS" else "tinjau",
            "size": len(idx), "sizeEstimated": False,
            "silhouette": sil, "silhouetteEstimated": False, "compactness": comp, "margin": mar,
            "gate": {"support": len(idx) >= t["support"], "silhouette": (sil or 0) >= t["silhouette"],
                     "compactness": comp is None or comp >= t["compactness"],
                     "margin": mar is None or mar >= t["margin"]},
            "bin": lab.get("bin", "").strip() or "karantina",
            "hazard": lab.get("hazard", "").strip() or "sedang",
            "visual": lab.get("visual", "").strip(),
            "components": [], "hazardNote": "",
            "steps": [s.strip() for s in lab.get("steps", "").split("|") if s.strip()],
            "note": "", "sources": [], "samples": samples,
        })
        print(f"  {cid}: {len(idx)} foto")

    trusted = sum(c["size"] for c in clusters if c["status"] == "lolos")
    review = sum(c["size"] for c in clusters if c["status"] == "tinjau")
    summary = {
        "meta": {
            "title": "Hasil klaster SiPilah", "demo": False,
            "source": f"SuperGallery {art.manifest.get('data_key', '')}",
            "images": {"unique": len(rows)},
            "clusters": {"total": len(clusters), "trusted": sum(c["status"] == "lolos" for c in clusters),
                         "review": sum(c["status"] == "tinjau" for c in clusters)},
            "coverage": {"clustered": trusted + review, "trusted": trusted, "review": review,
                         "noise": noise["size"] if noise else 0},
            "thresholds": t,
        },
        "families": prev.get("families", []),
        "bins": prev.get("bins", []),
        "sources": prev.get("sources", []),
        "clusters": clusters,
        "born": [],
        "noise": noise or {"id": "noise", "name": "Belum terkelompok", "family": "noise", "status": "karantina",
                           "size": 0, "bin": "karantina", "hazard": "sedang", "visual": "", "components": [],
                           "hazardNote": "", "steps": [], "note": "", "sources": [], "samples": []},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Selesai → {summary_path}\nContact sheet → {args.sheets}")
    if named < len(gal.codes):
        print(f"Isi kolom name/family/bin/hazard di {args.labels.name} lalu jalankan ulang.\n"
              f"  family: {FAMILIES}\n  bin   : {BINS}\n  hazard: tinggi | sedang | rendah\n"
              f"  steps : pisahkan langkah dengan ' | '")


if __name__ == "__main__":
    main()
