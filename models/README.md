# Models Directory

This directory stores trained machine learning models.

## Model Files

Trained models are saved with `.pkl` extension using joblib.

Example:
```
models/
├── xgboost_bvsp_2024.pkl
├── lightgbm_petr4_2024.pkl
└── random_forest_vale3_2024.pkl
```

## Saving Models

Use the training pipeline to save models:

```python
from src.training.train_pipeline import TrainingPipeline

pipeline = TrainingPipeline(
    symbol="^BVSP",
    start_date="2022-01-01",
    end_date="2024-12-31",
    model_type="xgboost"
)

results = pipeline.run()
pipeline.save_model("models/my_model.pkl")
```

Or use the command-line:

```bash
python train.py --symbol ^BVSP --save-model models/bvsp_model.pkl
```

## Loading Models

Load trained models for inference:

```python
from src.models.classifier import TradingClassifier

model = TradingClassifier.load_model("models/my_model.pkl", "xgboost")
predictions = model.predict(X)
```

## Model Versioning

Consider adding dates or versions to model filenames:
- `xgboost_bvsp_20241015.pkl`
- `lightgbm_petr4_v2.pkl`
- `random_forest_vale3_latest.pkl`

## Git Ignore

Model files are ignored by git (see `.gitignore`). To share models:
1. Use MLflow for experiment tracking
2. Upload to cloud storage (S3, GCS, Azure Blob)
3. Use DVC for data/model versioning
4. Share via releases with specific versions
