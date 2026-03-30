from __future__ import annotations

from kc_housing import (
    MODEL_SELECTION_FOLDS_PATH,
    MODEL_SELECTION_RESULTS_PATH,
    WINNER_PATH,
    load_data,
    prepare_data,
    run_model_selection,
)


def main() -> None:
    train_df, future_df, zipdemo_df = load_data()
    prepared = prepare_data(train_df, future_df, zipdemo_df)
    summary, _ = run_model_selection(prepared)

    winner = summary.iloc[0]
    print(
        f"[winner] {winner['Model']} "
        f"rmse=${winner['RMSE ($)']:,.0f} "
        f"mae=${winner['MAE ($)']:,.0f} "
        f"mape={winner['MAPE (%)']:.2f}%"
    )
    print(f"[artifacts] {MODEL_SELECTION_RESULTS_PATH}")
    print(f"[artifacts] {MODEL_SELECTION_FOLDS_PATH}")
    print(f"[artifacts] {WINNER_PATH}")


if __name__ == "__main__":
    main()
