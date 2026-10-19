# SiPilah backend

FastAPI kecil untuk demo: menerima foto barang, mengenali kelompoknya memakai hasil
notebook **SuperGallery** (DINOv3 + SigLIP2), lalu membalas kelompok, skor, dan
5 kelompok paling mirip. Sekaligus menyajikan frontend `sipilah-fe` di satu alamat.

```
foto → putar EXIF → potong area barang (Canny, sama dengan notebook) → embedding DINOv3 & SigLIP2
     → skor gabungan 0,55 : 0,45 ke tiap foto galeri → skor tertinggi per kelompok
     → dikenali kalau skor ≥ TAU_ACCEPT dan selisih #1−#2 ≥ TAU_MARGIN, selain itu "belum" (Wadah Periksa)
```

Tidak ada database: galeri dimuat dari `.npy` + CSV ke RAM saat nyala, feedback ditulis ke `data/feedback.jsonl`.

## Pasang (sekali)

```bash
cd sipilah-be
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10–3.12
# torch: pilih salah satu
pip install torch --index-url https://download.pytorch.org/whl/cpu   # laptop tanpa GPU NVIDIA
pip install torch                                                     # laptop dengan GPU NVIDIA
pip install -r requirements.txt
cp .env.example .env
```

## Pasang model dari temanmu

1. Ekstrak `SuperGallery_All_Artifacts.zip` ke `sipilah-be/artifacts/` (struktur di dalamnya biarkan apa adanya).
   Yang dipakai: `models/dinov3`, `models/siglip2`, `<kode>/retrieval_export/`, `<kode>/supergallery_final/`,
   `<kode>/roi/`, `<kode>/data_split.csv`. Folder lain (cache, embedding, clustering) boleh dihapus.
2. Cek sambungan — embedding backend harus sama dengan notebook:
   ```bash
   python scripts/check_parity.py
   ```
3. Setel ambang. Tahap 1 cepat tanpa model, tahap 2 mengecek foto uji dan memilih foto contoh:
   ```bash
   python scripts/tune_threshold.py
   python scripts/tune_threshold.py --check --accept 0.62 --margin 0.04 --examples 6
   ```
   Isi `TAU_ACCEPT` dan `TAU_MARGIN` di `.env`, buka `review/threshold_check.html` untuk dicek mata.
4. Isi data frontend:
   ```bash
   python scripts/export_fe.py
   ```
   Lalu isi `cluster_labels.csv` (nama, family, bin, hazard, steps) dengan melihat
   `review/contact_sheets/<ID>.jpg`, dan jalankan `export_fe.py` sekali lagi.

## Jalankan

```bash
cd ../sipilah-fe && npm run build && cd ../sipilah-be
uvicorn app.main:app --env-file .env --port 8000
```

Buka **http://localhost:8000**. Kamera browser hanya jalan di `localhost` atau HTTPS, jadi
buka dari laptop yang menjalankan backend.

Kalau artefak/model gagal dimuat, server tetap nyala: `/health` membalas `ok: false` beserta
alasannya dan FE otomatis berjalan dalam mode simulasi.

## Endpoint

| | Path | Isi |
|---|---|---|
| GET | `/health` | status siap, encoder, jumlah kelompok/foto galeri, ambang aktif |
| POST | `/infer?source=photo\|live` | multipart `image` → `{objects:[{bbox, status, cluster, score, margin, reason, gate, top}], latency_ms, duplicate}` |
| POST | `/feedback` | `{item_id, cluster, score, source, image?}` → `data/feedback.jsonl` (+ JPG) |
| GET | `/` | build FE (`FE_DIST`) |

`status` = `dikenali` atau `belum`. `reason` untuk `belum`: `low_score` (kurang mirip) atau
`ambiguous` (mirip dua kelompok sekaligus). `gate` = status asli notebook (KNOWN/REVIEW/UNKNOWN).

## Develop & tes tanpa model

```bash
python scripts/make_fake_artifacts.py --images ../sipilah-fe/public/images/sample --out artifacts_fake
ARTIFACTS_DIR=artifacts_fake ENCODER=fake uvicorn app.main:app --port 8000
python -m pytest -q
```

## Struktur

```
app/settings.py   baca .env
app/vision.py     buka foto, roi_crop & letterbox (salinan notebook), encoder HF / tiruan
app/engine.py     cari & muat artefak, galeri, skor per kelompok, keputusan, cegah dobel
app/schemas.py    bentuk JSON
app/main.py       FastAPI + sajikan FE
scripts/          make_fake_artifacts, check_parity, tune_threshold, export_fe
```
