from fastapi.testclient import TestClient

from api import app


class DummyModel:
    def predict(self, features):
        _ = features
        return [0]

    def predict_proba(self, features):
        _ = features
        return [[0.8, 0.2]]


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_endpoint():
    app.state.model = DummyModel()
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Bad experience"})
        assert response.status_code == 200
        body = response.json()
        assert body["label"] in ("positive", "negative")
        assert isinstance(body["confidence"], float)
