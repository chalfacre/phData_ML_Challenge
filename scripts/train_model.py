from __future__ import annotations

from pathlib import Path

from app.ml import load_data, save_bundle, train_model


def main() -> None:
    train_df, _, zip_demo = load_data()
    bundle = train_model(train_df, zip_demo)

    out = Path("model/model_bundle.joblib")
    save_bundle(bundle, out)

    print(f"saved bundle -> {out}")
    print(f"feature_count={len(bundle.feature_columns)}")
    print(f"model_version={bundle.version}")


if __name__ == "__main__":
    main()
