from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"


def main() -> None:
    future = pd.read_csv("data/future_unseen_examples.csv")

    sample = future.head(5).to_dict(orient="records")
    response = requests.post(API_URL, json={"records": sample}, timeout=30)
    response.raise_for_status()

    print("/predict response:")
    print(json.dumps(response.json(), indent=2))

    out = Path("output/test_api_response.json")
    out.write_text(json.dumps({"predict": response.json()}, indent=2))
    print(f"saved response -> {out}")


if __name__ == "__main__":
    main()
