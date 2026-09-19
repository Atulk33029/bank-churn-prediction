"""Load the raw bank churn dataset and turn it into model-ready features."""
import pandas as pd

from config import RAW_DATA, DROP_COLS, CATEGORICAL, TARGET


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    """Read the Kaggle Churn_Modelling.csv file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Download 'Churn Modelling' from Kaggle and place Churn_Modelling.csv "
            "in the data/ folder. See data/README.md for the link."
        )
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns that carry no predictive signal."""
    cols = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """A few simple engineered features that help the model."""
    out = df.copy()
    out["BalancePerProduct"] = out["Balance"] / out["NumOfProducts"].replace(0, 1)
    out["ZeroBalance"] = (out["Balance"] == 0).astype(int)
    out["TenurePerAge"] = out["Tenure"] / out["Age"]
    return out


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Geography and Gender."""
    cats = [c for c in CATEGORICAL if c in df.columns]
    return pd.get_dummies(df, columns=cats, drop_first=True)


def build_features(df: pd.DataFrame):
    """Full pipeline: clean -> engineer -> encode. Returns X, y."""
    work = encode(add_features(clean(df)))
    y = work[TARGET]
    X = work.drop(columns=[TARGET])
    return X, y


if __name__ == "__main__":
    raw = load_raw()
    X, y = build_features(raw)
    print(f"Rows: {len(raw):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Churn rate: {y.mean():.2%}")
    print(X.columns.tolist())
