# phData MLE Project - Sound Realty Housing Price API

This repository implements a complete solution for the ML engineering challenge:

- FastAPI REST service for housing price predictions
- Backend join to zipcode demographic data
- Dockerized deployment
- Test script that calls endpoints with provided future examples
- Model evaluation and basic model improvements
- Business + technical presentation outlines

## 1) Project Structure

- `app/main.py` - FastAPI app and endpoints
- `app/ml.py` - model training, feature engineering, evaluation, inference helpers
- `scripts/train_model.py` - trains and saves model bundle
- `scripts/evaluate_model.py` - baseline vs improved model metrics
- `scripts/test_api.py` - submits sample records from future examples
- `experiments/` - exploratory model-selection scripts (not part of serving path)

## 2) API Endpoints

### `GET /health`
Simple health endpoint.

### `POST /predict`
Input payload must contain all columns from `data/future_unseen_examples.csv`.

```json
{
  "records": [
    {
      "bedrooms": 3,
      "bathrooms": 2,
      "sqft_living": 1800,
      "sqft_lot": 5000,
      "floors": 1,
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
      "sqft_lot15": 5100
    }
  ]
}
```

## 3) Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train_model
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In another shell:

```bash
source .venv/bin/activate
python -m scripts.test_api
```

## 4) Docker Run (pre-trained artifact workflow)

Train model locally first (outside Docker):

```bash
python -m scripts.train_model
```

Then build and run container (which copies the saved artifact):

```bash
docker build -t sound-realty-api .
docker run --rm -p 8000:8000 sound-realty-api
```

Then test:

```bash
python -m scripts.test_api
```

If `/health` returns `"model_loaded": false`, confirm `model/model_bundle.joblib` exists before building.

## 5) Model Evaluation

```bash
python -m scripts.evaluate_model
```

Results are saved to `output/model_evaluation.json`.

## 6) Notes on Improvements

Baseline in this solution approximates a simple model using core house numeric features only.
Improved model adds:
- zipcode demographics joined in backend
- engineered features (`home_age`, `renovated`, ratios, log lot size)
- stronger tree-based regressor and cross-validation

## 7) Developer Tooling (Readability + Quality)

```bash
pip install -r requirements-dev.txt
ruff check app scripts tests
black --check app scripts tests
pytest
```

## 8) Production Considerations

See `docs/scaling_and_model_update_strategy.md` for:
- horizontal scaling approach
- no-downtime model update pattern
- recommended monitoring and MLOps controls
