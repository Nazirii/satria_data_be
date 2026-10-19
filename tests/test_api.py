import io
import json

from PIL import Image

from conftest import SAMPLES


def _jpg(path=None, size=(64, 64)):
    if path:
        return path.read_bytes()
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 140, 160)).save(buf, "JPEG")
    return buf.getvalue()


def test_health(client):
    h = client.get("/health").json()
    assert h["ok"] and h["mode"] == "fake"
    assert h["gallery"]["clusters"] > 1 and h["thresholds"]["source"] == "env"
    assert h["encoders"] == ["dinov3", "siglip2"]


def test_infer_known_sample(client):
    r = client.post("/infer", files={"image": ("a.jpg", _jpg(SAMPLES / "k21-3.jpg"), "image/jpeg")})
    assert r.status_code == 200
    o = r.json()["objects"][0]
    assert o["status"] == "dikenali" and o["cluster"].startswith("K")
    assert len(o["top"]) == 5 and o["top"][0]["cluster"] == o["cluster"]
    assert all(0 <= v <= 1 for v in o["bbox"]) and o["reason"] is None


def test_infer_rejects_non_image(client):
    r = client.post("/infer", files={"image": ("a.txt", b"bukan gambar", "text/plain")})
    assert r.status_code == 400


def test_infer_too_large(client):
    big = b"\xff" * (16 * 1024 * 1024)
    r = client.post("/infer", files={"image": ("a.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_live_duplicate(client):
    img = _jpg(SAMPLES / "k11-2.jpg")
    a = client.post("/infer?source=live", files={"image": ("a.jpg", img, "image/jpeg")}).json()
    b = client.post("/infer?source=live", files={"image": ("a.jpg", img, "image/jpeg")}).json()
    c = client.post("/infer?source=photo", files={"image": ("a.jpg", img, "image/jpeg")}).json()
    assert not a["duplicate"] and b["duplicate"] and not c["duplicate"]


def test_feedback_saved(client):
    import app.main as m

    r = client.post("/feedback", json={"item_id": "L-1", "cluster": "K03", "score": 0.7, "source": "live",
                                       "image": "data:image/jpeg;base64,/9j/2wBD"})
    assert r.json() == {"ok": True}
    line = (m.settings.data_dir / "feedback.jsonl").read_text().strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["item_id"] == "L-1" and rec["image_file"].endswith(".jpg")
