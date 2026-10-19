import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.engine import Gallery, Thresholds, decide  # noqa: E402
from app.vision import bbox_rel, letterbox, roi_crop  # noqa: E402


def test_roi_crop_finds_object():
    im = Image.new("RGB", (800, 600), (235, 235, 235))
    ImageDraw.Draw(im).rectangle((300, 200, 500, 400), fill=(30, 30, 30))
    crop, (x1, y1, x2, y2) = roi_crop(im)
    assert x1 <= 300 and y1 <= 200 and x2 >= 500 and y2 >= 400
    assert (x2 - x1) < 400 and (y2 - y1) < 350  # tidak memakan seluruh gambar
    assert crop.size == (x2 - x1, y2 - y1)


def test_roi_crop_blank_returns_whole_image():
    im = Image.new("RGB", (320, 200), (200, 200, 200))
    _, box = roi_crop(im)
    assert box == (0, 0, 320, 200)


def test_letterbox_size_and_padding():
    out = letterbox(Image.new("RGB", (400, 100), (255, 0, 0)), 224)
    assert out.size == (224, 224)
    assert out.getpixel((112, 2)) == (128, 128, 128)
    assert out.getpixel((112, 112)) == (255, 0, 0)


def test_bbox_rel():
    assert bbox_rel((10, 20, 60, 120), (100, 200)) == [0.1, 0.1, 0.5, 0.5]


def _gallery():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((9, 16)).astype(np.float32)
    rows = [{"sample_id": f"s{i}", "cluster_id": c} for i, c in enumerate([2, 0, 0, 1, 2, -1, 1, 0, 2])]
    return Gallery.build(rows, X, X[:, ::-1].copy(), 0.55), X


def test_cluster_scores_match_naive_loop():
    g, _ = _gallery()
    q = np.random.default_rng(2).standard_normal((3, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    got = g.cluster_scores(q, q[:, ::-1])
    s = g.similarity(q, q[:, ::-1])
    for c in range(len(g.codes)):
        np.testing.assert_allclose(got[:, c], s[:, g.row_cluster == c].max(axis=1), rtol=1e-6)
    assert g.codes == ["K00", "K01", "K02"] and len(g.rows) == 8  # noise dibuang


def test_exclude_self():
    g, _ = _gallery()
    q = g.primary[:1]
    full = g.cluster_scores(q, g.secondary[:1])[0]
    excl = g.cluster_scores(q, g.secondary[:1], exclude=np.array([0]))[0]
    assert full[g.row_cluster[0]] > 0.999
    assert excl[g.row_cluster[0]] < full[g.row_cluster[0]]


def test_decide():
    t = Thresholds(accept=0.6, margin=0.05, unknown=0.4, source="t")
    assert decide(0.8, 0.5, t) == ("dikenali", None, "KNOWN")
    assert decide(0.8, 0.78, t) == ("belum", "ambiguous", "REVIEW")
    assert decide(0.5, 0.2, t) == ("belum", "low_score", "REVIEW")
    assert decide(0.3, 0.1, t) == ("belum", "low_score", "UNKNOWN")
