"""Central paths and settings for the churn project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "Churn_Modelling.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_rf_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
POWERBI_DIR = ROOT / "powerbi"
POWERBI_EXPORT = POWERBI_DIR / "churn_predictions.csv"

TARGET = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]
CATEGORICAL = ["Geography", "Gender"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

for _d in (MODEL_DIR, FIGURES_DIR, POWERBI_DIR):
    _d.mkdir(parents=True, exist_ok=True)
