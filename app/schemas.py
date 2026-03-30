from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FullPredictionRow(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    zipcode: str | int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float


class PredictionRequest(BaseModel):
    records: list[FullPredictionRow] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    model_version: str
    prediction_count: int
    predictions: list[float]
    metadata: dict[str, Any]
