from __future__ import annotations

import json
from pathlib import Path

from app.ml import evaluate_model, load_data


def main() -> None:
    train_df, _, zip_demo = load_data()
    metrics = evaluate_model(train_df, zip_demo)

    out = Path("output/model_evaluation.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
