from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import RuntimeState, app
from app.ml import FULL_INPUT_COLUMNS


class DummyModel:
    def predict(self, X):  # noqa: N803
        return np.full(shape=(len(X),), fill_value=123456.0)


def _full_record() -> dict:
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "yr_built": 1995,
        "yr_renovated": 0,
        "zipcode": "98001",
        "lat": 47.3,
        "long": -122.2,
        "sqft_living15": 1700,
        "sqft_lot15": 5100,
    }


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_endpoint_with_mock_runtime() -> None:
    runtime = RuntimeState()
    runtime.bundle = type(
        "Bundle",
        (),
        {"model": DummyModel(), "feature_columns": FULL_INPUT_COLUMNS, "version": "test"},
    )()
    runtime.zip_demo = pd.DataFrame([{"zipcode": "98001"}])
    app.state.runtime = runtime

    with TestClient(app) as client:
        response = client.post("/predict", json={"records": [_full_record()]})
        assert response.status_code == 200
        payload = response.json()
        assert payload["prediction_count"] == 1
        assert len(payload["predictions"]) == 1
        assert isinstance(payload["predictions"][0], float)
