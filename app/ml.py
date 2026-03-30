from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "kc_house_data.csv"
FUTURE_PATH = DATA_DIR / "future_unseen_examples.csv"
ZIP_PATH = DATA_DIR / "zipcode_demographics.csv"

FULL_INPUT_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 24,
    "min_samples_leaf": 2,
    "max_features": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class ModelBundle:
    model: RandomForestRegressor
    feature_columns: list[str]
    version: str


def build_model() -> RandomForestRegressor:
    """Create the primary regressor with centrally managed hyperparameters."""
    return RandomForestRegressor(**MODEL_PARAMS)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    future = pd.read_csv(FUTURE_PATH)
    zip_demo = pd.read_csv(ZIP_PATH)
    return train, future, zip_demo


def enrich_and_engineer(df: pd.DataFrame, zip_demo: pd.DataFrame) -> pd.DataFrame:
    """Join zipcode demographics and derive lightweight predictive features."""
    frame = df.copy(deep=True).assign(zipcode=lambda d: d["zipcode"].astype(str))
    zip_copy = zip_demo.copy(deep=True).assign(zipcode=lambda d: d["zipcode"].astype(str))

    merged = frame.merge(zip_copy, on="zipcode", how="left")

    merged = merged.assign(
        home_age=2015 - merged["yr_built"],
        renovated=(merged["yr_renovated"] > 0).astype(int),
        living_per_bedroom=merged["sqft_living"] / merged["bedrooms"].clip(lower=1),
    )

    for col in ("sqft_lot", "sqft_lot15"):
        if col in merged.columns:
            merged = merged.assign(**{f"{col}_log1p": np.log1p(merged[col].clip(lower=0))})

    return merged


def train_model(train_df: pd.DataFrame, zip_demo: pd.DataFrame) -> ModelBundle:
    """Train the production model and return model metadata for serving."""
    engineered = enrich_and_engineer(train_df, zip_demo)
    X = engineered.drop(columns=["id", "date", "price"], errors="ignore")
    y = engineered["price"]

    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    model = build_model()
    model.fit(X, y)

    return ModelBundle(model=model, feature_columns=list(X.columns), version="1.0.0")


def evaluate_model(train_df: pd.DataFrame, zip_demo: pd.DataFrame) -> dict[str, float]:
    """Compare baseline vs improved feature sets using 5-fold cross-validation."""
    engineered = enrich_and_engineer(train_df, zip_demo)

    baseline_numeric = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]

    X_base = engineered[baseline_numeric].fillna(0)
    X_improved = engineered.drop(columns=["id", "date", "price"], errors="ignore")
    X_improved = X_improved.fillna(X_improved.median(numeric_only=True)).fillna(0)
    y = engineered["price"]

    splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    def cv_rmse_mae(X: pd.DataFrame) -> tuple[float, float, float]:
        cv = cross_validate(
            build_model(),
            X,
            y,
            cv=splitter,
            scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"),
            n_jobs=-1,
        )
        rmse = float(-cv["test_neg_root_mean_squared_error"].mean())
        mae = float(-cv["test_neg_mean_absolute_error"].mean())
        r2 = float(cv["test_r2"].mean())
        return rmse, mae, r2

    base_rmse, base_mae, base_r2 = cv_rmse_mae(X_base)
    imp_rmse, imp_mae, imp_r2 = cv_rmse_mae(X_improved)

    return {
        "baseline_rmse": base_rmse,
        "baseline_mae": base_mae,
        "baseline_r2": base_r2,
        "improved_rmse": imp_rmse,
        "improved_mae": imp_mae,
        "improved_r2": imp_r2,
    }


def save_bundle(bundle: ModelBundle, path: Path = MODEL_DIR / "model_bundle.joblib") -> None:
    payload: dict[str, Any] = {
        "model": bundle.model,
        "feature_columns": bundle.feature_columns,
        "version": bundle.version,
    }
    joblib.dump(payload, path)


def load_bundle(path: Path = MODEL_DIR / "model_bundle.joblib") -> ModelBundle:
    payload = joblib.load(path)
    return ModelBundle(
        model=payload["model"],
        feature_columns=payload["feature_columns"],
        version=payload["version"],
    )


def prepare_features(
    records: list[dict[str, Any]],
    zip_demo: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform request payloads into the exact feature matrix expected by the model."""
    frame = pd.DataFrame(records)
    engineered = enrich_and_engineer(frame, zip_demo)
    X = engineered.reindex(columns=feature_columns, fill_value=np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X
