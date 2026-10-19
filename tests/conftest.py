import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT.parent / "sipilah-fe" / "public" / "images" / "sample"


@pytest.fixture(scope="session")
def fake_artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_artifacts.py"), "--images", str(SAMPLES),
                    "--out", str(out), "--views", "4"], check=True, capture_output=True)
    return out


@pytest.fixture(scope="session")
def client(fake_artifacts, tmp_path_factory):
    os.environ.update({"ARTIFACTS_DIR": str(fake_artifacts), "ENCODER": "fake", "FE_DIST": "/nonexistent",
                       "DATA_DIR": str(tmp_path_factory.mktemp("data")), "TAU_ACCEPT": "0.6", "TAU_MARGIN": "0.02"})
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
