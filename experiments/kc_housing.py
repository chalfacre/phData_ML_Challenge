from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import time
import warnings

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MPL_CACHE_DIR = OUTPUT_DIR / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = ROOT / "data"
MODEL_PATH = OUTPUT_DIR / "final_model.pkl"
MODEL_METADATA_PATH = OUTPUT_DIR / "final_model_metadata.json"
WINNER_PATH = OUTPUT_DIR / "winner_model.json"
PREDICTIONS_PATH = OUTPUT_DIR / "future_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
XGBOOST_CV_RESULTS_PATH = OUTPUT_DIR / "xgboost_cv_results.csv"
XGBOOST_CV_FOLDS_PATH = OUTPUT_DIR / "xgboost_cv_fold_results.csv"
MODEL_SELECTION_RESULTS_PATH = OUTPUT_DIR / "model_selection_results.csv"
MODEL_SELECTION_FOLDS_PATH = OUTPUT_DIR / "model_selection_fold_results.csv"
LEGACY_MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison_results.csv"


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    y_train_log: pd.Series
    X_future: pd.DataFrame
    future_raw: pd.DataFrame
    preprocessor: "HousingPreprocessor"


@dataclass
class HousingPreprocessor:
    reference_sale_year: int | None = None
    reference_sale_month: int | None = None
    zipcode_map: dict[str, int] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    medians: pd.Series | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        zipdemo_df: pd.DataFrame,
    ) -> "HousingPreprocessor":
        sale_dates = pd.to_datetime(
            train_df["date"],
            format="%Y%m%dT%H%M%S",
            errors="coerce",
        )
        sale_years = sale_dates.dt.year.dropna()
        sale_months = sale_dates.dt.month.dropna()

        self.reference_sale_year = int(sale_years.mode().iloc[0]) if not sale_years.empty else 2014
        self.reference_sale_month = int(sale_months.mode().iloc[0]) if not sale_months.empty else 6

        zipcode_values = sorted(
            set(train_df["zipcode"].astype(str))
            | set(future_df["zipcode"].astype(str))
            | set(zipdemo_df["zipcode"].astype(str))
        )
        self.zipcode_map = {zipcode: idx for idx, zipcode in enumerate(zipcode_values)}

        engineered_train = self._engineer(train_df, zipdemo_df)
        X_train = engineered_train.drop(columns=["price"], errors="ignore")

        self.feature_columns = list(X_train.columns)
        self.medians = X_train.median(numeric_only=True)
        return self

    def transform(self, df: pd.DataFrame, zipdemo_df: pd.DataFrame) -> pd.DataFrame:
        engineered = self._engineer(df, zipdemo_df)
        X = engineered.drop(columns=["price"], errors="ignore")
        X = X.reindex(columns=self.feature_columns, fill_value=np.nan)
        X = X.fillna(self.medians)
        return X.fillna(0.0)

    def _engineer(self, df: pd.DataFrame, zipdemo_df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        zipdemo = zipdemo_df.copy()
        zipdemo["zipcode"] = zipdemo["zipcode"].astype(str)

        frame["zipcode"] = frame["zipcode"].astype(str)

        if "date" in frame.columns:
            sale_dates = pd.to_datetime(
                frame["date"],
                format="%Y%m%dT%H%M%S",
                errors="coerce",
            )
            frame["sale_year"] = sale_dates.dt.year.fillna(self.reference_sale_year)
            frame["sale_month"] = sale_dates.dt.month.fillna(self.reference_sale_month)
            frame["sale_quarter"] = sale_dates.dt.quarter.fillna((self.reference_sale_month - 1) // 3 + 1)
            frame = frame.drop(columns=["date"])
        else:
            frame["sale_year"] = self.reference_sale_year
            frame["sale_month"] = self.reference_sale_month
            frame["sale_quarter"] = (self.reference_sale_month - 1) // 3 + 1

        frame["bedrooms"] = frame["bedrooms"].clip(upper=10)
        frame["sqft_lot"] = frame["sqft_lot"].clip(lower=0)
        frame["sqft_lot15"] = frame["sqft_lot15"].clip(lower=0)

        frame["sqft_lot_log1p"] = np.log1p(frame["sqft_lot"])
        frame["sqft_lot15_log1p"] = np.log1p(frame["sqft_lot15"])

        frame["renovation_reference_year"] = np.where(
            frame["yr_renovated"].eq(0),
            frame["yr_built"],
            frame["yr_renovated"],
        )
        frame["house_age"] = frame["sale_year"] - frame["yr_built"]
        frame["years_since_renovation"] = frame["sale_year"] - frame["renovation_reference_year"]
        frame["was_renovated"] = frame["yr_renovated"].gt(0).astype(int)
        frame["basement_flag"] = frame["sqft_basement"].gt(0).astype(int)
        frame["total_rooms"] = frame["bedrooms"] + frame["bathrooms"]
        frame["bathrooms_per_bedroom"] = frame["bathrooms"] / frame["bedrooms"].clip(lower=1)
        frame["living_to_lot_ratio"] = frame["sqft_living"] / frame["sqft_lot"].clip(lower=1)
        frame["living_to_neighbor_ratio"] = frame["sqft_living"] / frame["sqft_living15"].clip(lower=1)
        frame["grade_living_interaction"] = frame["grade"] * frame["sqft_living"]

        frame = frame.merge(zipdemo, on="zipcode", how="left")
        frame["zipcode"] = frame["zipcode"].map(self.zipcode_map).fillna(-1).astype(int)

        drop_columns = ["id", "sqft_lot", "sqft_lot15"]
        return frame.drop(columns=[column for column in drop_columns if column in frame.columns])


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "kc_house_data.csv")
    future = pd.read_csv(data_dir / "future_unseen_examples.csv")
    zipdemo = pd.read_csv(data_dir / "zipcode_demographics.csv")
    print(f"[load] train={train.shape} future={future.shape} zipdemo={zipdemo.shape}")
    return train, future, zipdemo


def prepare_data(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    zipdemo_df: pd.DataFrame,
) -> PreparedData:
    preprocessor = HousingPreprocessor().fit(train_df, future_df, zipdemo_df)
    X_train = preprocessor.transform(train_df, zipdemo_df)
    X_future = preprocessor.transform(future_df, zipdemo_df)
    y_train_log = np.log1p(train_df["price"])

    print(
        f"[prepare] X_train={X_train.shape} X_future={X_future.shape} "
        f"features={X_train.shape[1]}"
    )
    return PreparedData(
        X_train=X_train,
        y_train_log=y_train_log,
        X_future=X_future,
        future_raw=future_df.copy(),
        preprocessor=preprocessor,
    )


def build_xgboost_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=900,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.03,
        reg_lambda=1.2,
        gamma=0.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )


def build_model_candidates() -> dict[str, object]:
    return {
        "XGBoost": build_xgboost_model(),
        "LightGBM": lgb.LGBMRegressor(
            objective="regression",
            n_estimators=700,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=48,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=24,
            min_samples_leaf=2,
            max_features=0.65,
            n_jobs=-1,
            random_state=42,
        ),
    }


def get_model_by_name(model_name: str) -> object:
    candidates = build_model_candidates()
    if model_name not in candidates:
        raise ValueError(f"Unsupported model: {model_name}")
    return candidates[model_name]


def regression_metrics(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    actual = np.expm1(y_true_log)
    predicted = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)

    return {
        "MAE ($)": mean_absolute_error(actual, predicted),
        "RMSE ($)": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE (%)": mean_absolute_percentage_error(actual, predicted) * 100.0,
        "R²": r2_score(actual, predicted),
    }


def evaluate_models(
    models: dict[str, object],
    X: pd.DataFrame,
    y_log: pd.Series,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    summary_records: list[dict[str, float | str]] = []
    fold_records: list[dict[str, float | str | int]] = []

    for model_name, model in models.items():
        print(f"[cv] {model_name}")
        model_fold_metrics: list[dict[str, float]] = []
        started_at = time.time()

        for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            estimator = clone(model)

            X_train_fold = X.iloc[train_idx]
            X_valid_fold = X.iloc[valid_idx]
            y_train_fold = y_log.iloc[train_idx]
            y_valid_fold = y_log.iloc[valid_idx]

            estimator.fit(X_train_fold, y_train_fold)
            valid_pred_log = estimator.predict(X_valid_fold)
            fold_metrics = regression_metrics(y_valid_fold, valid_pred_log)
            model_fold_metrics.append(fold_metrics)

            fold_records.append(
                {
                    "Model": model_name,
                    "Fold": fold_number,
                    **fold_metrics,
                }
            )

            print(
                f"    fold={fold_number} "
                f"rmse=${fold_metrics['RMSE ($)']:,.0f} "
                f"mae=${fold_metrics['MAE ($)']:,.0f} "
                f"mape={fold_metrics['MAPE (%)']:.2f}% "
                f"r2={fold_metrics['R²']:.4f}"
            )

        elapsed = time.time() - started_at
        metrics_frame = pd.DataFrame(model_fold_metrics)
        summary_records.append(
            {
                "Model": model_name,
                "MAE ($)": metrics_frame["MAE ($)"].mean(),
                "MAE Std": metrics_frame["MAE ($)"].std(ddof=0),
                "RMSE ($)": metrics_frame["RMSE ($)"].mean(),
                "RMSE Std": metrics_frame["RMSE ($)"].std(ddof=0),
                "MAPE (%)": metrics_frame["MAPE (%)"].mean(),
                "R²": metrics_frame["R²"].mean(),
                "Train Time (s)": round(elapsed, 1),
            }
        )

    summary = pd.DataFrame(summary_records).sort_values(
        by=["RMSE ($)", "MAE ($)", "MAPE (%)"],
        ascending=[True, True, True],
    )
    folds = pd.DataFrame(fold_records)
    return summary, folds


def train_final_model(model_name: str, X: pd.DataFrame, y_log: pd.Series) -> object:
    model = get_model_by_name(model_name)
    model.fit(X, y_log)
    return model


def save_model_artifacts(model: object, model_name: str, feature_columns: list[str]) -> None:
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_name": model_name,
        "model_path": str(MODEL_PATH),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"[save] model -> {MODEL_PATH}")
    print(f"[save] model metadata -> {MODEL_METADATA_PATH}")


def predict_prices(model: object, X_future: pd.DataFrame) -> np.ndarray:
    predictions_log = model.predict(X_future)
    predictions = np.clip(np.expm1(predictions_log), a_min=0, a_max=None)
    return predictions


def save_predictions(future_df: pd.DataFrame, predictions: np.ndarray, path: Path = PREDICTIONS_PATH) -> None:
    output = future_df.copy()
    output["predicted_price"] = np.rint(predictions).astype(int)
    output.to_csv(path, index=False)
    print(f"[save] predictions -> {path}")


def save_feature_importance(
    model: object,
    feature_names: list[str],
    path: Path = FEATURE_IMPORTANCE_PATH,
) -> None:
    if not hasattr(model, "feature_importances_"):
        print("[save] feature importance skipped: model has no feature_importances_")
        return

    importance_frame = (
        pd.Series(model.feature_importances_, index=feature_names, name="importance")
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index()
    )
    importance_frame.to_csv(path, index=False)
    print(f"[save] feature importance -> {path}")


def print_summary_table(summary: pd.DataFrame) -> None:
    print("\nModel summary (lower RMSE is better)")
    for _, row in summary.iterrows():
        print(
            f"  {row['Model']:<12} "
            f"rmse=${row['RMSE ($)']:>10,.0f} "
            f"mae=${row['MAE ($)']:>9,.0f} "
            f"mape={row['MAPE (%)']:.2f}% "
            f"r2={row['R²']:.4f} "
            f"time={row['Train Time (s)']:.1f}s"
        )


def run_model_selection(prepared: PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, folds = evaluate_models(
        build_model_candidates(),
        prepared.X_train,
        prepared.y_train_log,
        n_splits=5,
    )
    summary.to_csv(MODEL_SELECTION_RESULTS_PATH, index=False)
    summary.to_csv(LEGACY_MODEL_COMPARISON_PATH, index=False)
    folds.to_csv(MODEL_SELECTION_FOLDS_PATH, index=False)

    winner = summary.iloc[0]
    WINNER_PATH.write_text(
        json.dumps(
            {
                "model_name": winner["Model"],
                "rmse": float(winner["RMSE ($)"]),
                "mae": float(winner["MAE ($)"]),
                "mape": float(winner["MAPE (%)"]),
                "r2": float(winner["R²"]),
            },
            indent=2,
        )
    )

    print_summary_table(summary)
    print(f"[save] model selection summary -> {MODEL_SELECTION_RESULTS_PATH}")
    print(f"[save] legacy comparison summary -> {LEGACY_MODEL_COMPARISON_PATH}")
    print(f"[save] model selection folds -> {MODEL_SELECTION_FOLDS_PATH}")
    print(f"[save] winner metadata -> {WINNER_PATH}")
    return summary, folds


def main() -> None:
    train_df, future_df, zipdemo_df = load_data()
    prepared = prepare_data(train_df, future_df, zipdemo_df)

    summary, _ = run_model_selection(prepared)
    winner_model = str(summary.iloc[0]["Model"])
    print(f"[winner] selected model={winner_model}")

    final_model = train_final_model(winner_model, prepared.X_train, prepared.y_train_log)
    save_model_artifacts(final_model, winner_model, prepared.preprocessor.feature_columns)

    predictions = predict_prices(final_model, prepared.X_future)
    save_predictions(prepared.future_raw, predictions)
    save_feature_importance(final_model, prepared.preprocessor.feature_columns)

    print(
        f"[predict] min=${predictions.min():,.0f} "
        f"mean=${predictions.mean():,.0f} "
        f"max=${predictions.max():,.0f}"
    )


if __name__ == "__main__":
    main()
