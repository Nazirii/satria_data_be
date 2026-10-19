#!/usr/bin/env python3
"""Setel ambang "dikenali" untuk demo.

Tahap 1 (tanpa model, cepat): setiap foto berklaster disembunyikan lalu ditebak ulang
dari foto lain. Hasilnya tabel: kalau ambangnya x, berapa % foto dikenali dan berapa %
tebakannya benar. Dari situ keluar saran angka.

Tahap 2 (--check, pakai model dari .env): foto uji yang TIDAK ada di galeri ditebak
dengan ambang terpilih, lalu ditulis ke review/threshold_check.html untuk dicek mata.
Tambah --examples N untuk menyalin N foto contoh ke FE (tombol "Coba foto contoh").

  python scripts/tune_threshold.py
  python scripts/tune_threshold.py --check --accept 0.62 --margin 0.02 --examples 6
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.check_parity import load_env  # noqa: E402


def leave_one_out(art, gal):
    from app.engine import read_csv

    rows = read_csv(art.final_dir / "gallery_assignments.csv")
    Xp = np.load(art.final_dir / "development_embeddings_primary.npy").astype(np.float32)
    Xs = np.load(art.final_dir / "development_embeddings_siglip2.npy").astype(np.float32) if gal.fused else None
    Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
    if Xs is not None:
        Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    in_gal = {r["sample_id"]: i for i, r in enumerate(gal.rows)}
    t1, t2, correct, is_noise = [], [], [], []
    for a in range(0, len(rows), 512):
        b = min(len(rows), a + 512)
        excl = np.array([in_gal.get(r["sample_id"], -1) for r in rows[a:b]])
        S = gal.cluster_scores(Xp[a:b], None if Xs is None else Xs[a:b], exclude=excl)
        srt = np.sort(S, axis=1)
        t1 += srt[:, -1].tolist()
        t2 += (srt[:, -2] if S.shape[1] > 1 else np.zeros(len(S))).tolist()
        pred = np.argmax(S, axis=1)
        for k, r in enumerate(rows[a:b]):
            cid = int(float(r["cluster_id"]))
            is_noise.append(cid == -1)
            correct.append(cid != -1 and int(gal.cluster_ids[pred[k]]) == cid)
    return np.array(t1), np.array(t2), np.array(correct), np.array(is_noise)


def sweep(t1, t2, correct, noise, target):
    member = ~noise
    rows = []
    for m in (0.02, 0.04, 0.06):
        for a in np.round(np.arange(0.30, 0.96, 0.01), 2):
            acc_mask = (t1 >= a) & ((t1 - t2) >= m)
            n_acc = (acc_mask & member).sum()
            rows.append({
                "accept": float(a), "margin": m,
                "dikenali_%": 100 * n_acc / max(1, member.sum()),
                "benar_%": 100 * (acc_mask & correct).sum() / max(1, n_acc),
                "noise_diterima_%": 100 * (acc_mask & noise).sum() / max(1, noise.sum()),
            })
    ok = [r for r in rows if r["benar_%"] >= target and r["dikenali_%"] > 0]
    best = max(ok, key=lambda r: (r["dikenali_%"], -r["margin"])) if ok else None
    return rows, best


def img_b64(path, size=150):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def check(eng, accept, margin, n, out_html, examples, fe_dir, labels_csv):
    from app.engine import Thresholds, decide, read_csv
    from app.vision import read_rgb

    art, gal = eng.art, eng.gallery
    split = art.run_dir / "data_split.csv"
    if not split.exists():
        sys.exit("data_split.csv tidak ada; foto uji tidak bisa ditemukan.")
    test = [r for r in read_csv(split) if r.get("split") == "test" and (art.roi_dir / f"{r['pixel_hash']}.jpg").exists()]
    rng = np.random.default_rng(0)
    test = [test[i] for i in rng.permutation(len(test))[:n]]
    th = Thresholds(accept=accept, margin=margin, unknown=min(eng.thresholds.unknown, accept), source="cli")
    members = {}
    for i, r in enumerate(gal.rows):
        members.setdefault(gal.codes[gal.row_cluster[i]], []).append(r)
    names = {}
    if labels_csv.exists():
        with labels_csv.open(newline="", encoding="utf-8") as f:
            names = {r["cluster"]: r.get("name", "").strip() for r in csv.DictReader(f)}

    results = []
    for r in test:
        path = art.roi_dir / f"{r['pixel_hash']}.jpg"
        qp, qs = eng.embed(read_rgb(path))
        top, t1, t2 = eng.rank(qp, qs)
        status, reason, _ = decide(t1, t2, th)
        results.append((r, path, top, t1, t2, status, reason))

    n_ok = sum(x[5] == "dikenali" for x in results)
    parts = [f"<h1>Cek ambang: accept {accept}, margin {margin}</h1>",
             f"<p>{n_ok}/{len(results)} foto uji dikenali. Tiap baris: foto uji, lalu 4 foto galeri dari kelompok tebakan.</p>"]
    for r, path, top, t1, t2, status, reason in results:
        c = top[0]["cluster"]
        refs = "".join(f"<img src='{img_b64(art.roi_dir / (m['pixel_hash'] + '.jpg'), 110)}'>"
                       for m in members.get(c, [])[:4] if (art.roi_dir / (m['pixel_hash'] + '.jpg')).exists())
        label = f"{c} {html.escape(names.get(c, ''))}"
        parts.append(
            f"<div class='row {status}'><img class='q' src='{img_b64(path)}'><div class='t'><b>{status}</b>"
            f" · {label} · skor {t1:.3f} · selisih {t1 - t2:.3f}{' · ' + reason if reason else ''}</div>{refs}</div>")
    style = ("<style>body{font:14px system-ui;margin:24px;background:#f6f7f5}.row{display:flex;gap:8px;align-items:center;"
             "padding:8px;margin:6px 0;background:#fff;border-radius:8px}.row.belum{background:#fff6e5}.t{width:300px}"
             "img{border-radius:4px}.q{outline:3px solid #222}</style>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("<meta charset='utf-8'>" + style + "".join(parts), encoding="utf-8")
    print(f"{n_ok}/{len(results)} foto uji dikenali → {out_html}")

    if examples and fe_dir:
        from PIL import Image

        known = [x for x in results if x[5] == "dikenali"]
        known.sort(key=lambda x: -x[3])
        pick, seen = [], set()
        for x in known:
            if x[2][0]["cluster"] not in seen:
                pick.append(x)
                seen.add(x[2][0]["cluster"])
            if len(pick) >= max(1, examples - 1):
                break
        unknown = [x for x in results if x[5] == "belum"]
        if unknown:
            pick.append(unknown[0])
        dst = fe_dir / "public" / "images" / "examples"
        dst.mkdir(parents=True, exist_ok=True)
        for old in dst.glob("*.jpg"):
            old.unlink()
        items = []
        for k, (r, path, top, _t1, _t2, status, _reason) in enumerate(pick, 1):
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((900, 900))
                im.save(dst / f"contoh-{k}.jpg", quality=88)
            c = top[0]["cluster"]
            label = names.get(c) or f"Contoh {k}"
            items.append({"key": f"contoh-{k}", "src": f"images/examples/contoh-{k}.jpg",
                          "label": label if status == "dikenali" else "Barang asing"})
        (fe_dir / "public" / "data" / "examples.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
        print(f"{len(items)} foto contoh → {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=95.0, help="minimal %% tebakan benar di antara yang dikenali")
    ap.add_argument("--check", action="store_true", help="tahap 2: jalankan encoder ke foto uji")
    ap.add_argument("--accept", type=float)
    ap.add_argument("--margin", type=float)
    ap.add_argument("-n", type=int, default=60, help="jumlah foto uji di tahap 2")
    ap.add_argument("--examples", type=int, default=0, help="salin N foto contoh ke FE")
    ap.add_argument("--fe", type=Path, default=ROOT.parent / "sipilah-fe")
    ap.add_argument("--labels", type=Path, default=ROOT / "cluster_labels.csv")
    ap.add_argument("--env", type=Path, default=ROOT / ".env")
    args = ap.parse_args()
    load_env(args.env)

    from app.engine import find_artifacts, load_gallery
    from app.settings import load_settings

    s = load_settings()
    art = find_artifacts(s.artifacts_dir)
    gal = load_gallery(art, s.gallery)
    t1, t2, correct, noise = leave_one_out(art, gal)
    rows, best = sweep(t1, t2, correct, noise, args.target)

    print(f"Foto berklaster: {(~noise).sum()}, foto noise: {noise.sum()}, galeri: {s.gallery}\n")
    print(f"{'accept':>6} {'margin':>6} {'dikenali':>9} {'benar':>7} {'noise ikut':>10}")
    show = [r for r in rows if r["margin"] == (best["margin"] if best else 0.02) and round(r["accept"] * 100) % 5 == 0]
    for r in show:
        mark = "  ← saran" if best and r["accept"] == best["accept"] else ""
        print(f"{r['accept']:6.2f} {r['margin']:6.2f} {r['dikenali_%']:8.1f}% {r['benar_%']:6.1f}% {r['noise_diterima_%']:9.1f}%{mark}")
    if best:
        print(f"\nSaran (benar ≥ {args.target:g}%, dikenali terbanyak): accept {best['accept']:.2f}, margin {best['margin']:.2f}"
              f" → {best['dikenali_%']:.1f}% dikenali, {best['benar_%']:.1f}% benar")
        print(f"Isi di .env:\n  TAU_ACCEPT={best['accept']:.2f}\n  TAU_MARGIN={best['margin']:.2f}")
    else:
        print(f"\nTidak ada ambang dengan benar ≥ {args.target:g}%. Coba --target lebih rendah.")

    if args.check:
        from app.engine import Engine

        eng = Engine(s)
        if not eng.ready:
            sys.exit(f"Backend belum siap: {eng.error}")
        accept = args.accept if args.accept is not None else (best["accept"] if best else eng.thresholds.accept)
        margin = args.margin if args.margin is not None else (best["margin"] if best else eng.thresholds.margin)
        check(eng, accept, margin, args.n, ROOT / "review" / "threshold_check.html", args.examples, args.fe, args.labels)


if __name__ == "__main__":
    main()
