"""The four public endpoints on one client (toy artifact, not models/)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import application_payload


def test_all_four_endpoints_on_one_client(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/model/info").status_code == 200
    payload = application_payload()
    predict = client.post("/predict", json=payload)
    explain = client.post("/predict/explain", json=payload)
    assert predict.status_code == 200
    assert explain.status_code == 200
    assert explain.json()["p_bad"] == predict.json()["p_bad"]
    assert client.post("/predict", json=application_payload(age=12)).status_code == 422
