import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    payload = {
        "features": {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 181,
            "dst_bytes": 5450,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 8,
            "srv_count": 8,
            "serror_rate": 0,
            "srv_serror_rate": 0,
            "rerror_rate": 0,
            "srv_rerror_rate": 0,
            "same_srv_rate": 1,
            "diff_srv_rate": 0,
            "srv_diff_host_rate": 0,
            "dst_host_count": 9,
            "dst_host_srv_count": 9,
            "dst_host_same_srv_rate": 1,
            "dst_host_diff_srv_rate": 0,
            "dst_host_same_src_port_rate": 0.11,
            "dst_host_srv_diff_host_rate": 0,
            "dst_host_serror_rate": 0,
            "dst_host_srv_serror_rate": 0,
            "dst_host_rerror_rate": 0,
            "dst_host_srv_rerror_rate": 0
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["label"] in ["Normal", "Attack"]
    assert response.json()["prediction"] in [0, 1]