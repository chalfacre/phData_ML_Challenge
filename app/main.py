from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.ml import FULL_INPUT_COLUMNS, ZIP_PATH, ModelBundle, load_bundle, prepare_features
from app.schemas import PredictionRequest, PredictionResponse


class RuntimeState:
    bundle: ModelBundle | None = None
    zip_demo: pd.DataFrame | None = None


def _to_records(items: list[Any]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in items]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load long-lived runtime artifacts (model + lookup data) at app startup."""
    state = RuntimeState()
    bundle_path = Path("model/model_bundle.joblib")
    if bundle_path.exists():
        state.bundle = load_bundle(bundle_path)
        state.zip_demo = pd.read_csv(ZIP_PATH).copy(deep=True)
    app.state.runtime = state
    yield


app = FastAPI(title="Sound Realty Housing API", version="1.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | bool]:
    runtime: RuntimeState | None = getattr(app.state, "runtime", None)
    model_loaded = bool(runtime and runtime.bundle is not None and runtime.zip_demo is not None)
    return {"status": "ok", "model_loaded": model_loaded}


def _runtime() -> RuntimeState:
    runtime: RuntimeState | None = getattr(app.state, "runtime", None)
    if runtime is None or runtime.bundle is None or runtime.zip_demo is None:
        raise HTTPException(status_code=503, detail="Model runtime not loaded")
    return runtime


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    runtime = _runtime()
    records = _to_records(payload.records)

    X = prepare_features(records, runtime.zip_demo, runtime.bundle.feature_columns)
    predictions = runtime.bundle.model.predict(X)

    return PredictionResponse(
        model_version=runtime.bundle.version,
        prediction_count=len(predictions),
        predictions=[float(x) for x in predictions],
        metadata={
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "endpoint": "/predict",
            "demographics_joined_backend": True,
            "required_columns": FULL_INPUT_COLUMNS,
        },
    )
