from pathlib import Path
import pandas as pd
import yaml

def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=["date"])
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_marginal_prices(path: str | Path = "data/marginal_prices.parquet") -> pd.DataFrame:
    return _load_table(path)


def load_rem_offers(path: str | Path = "data/rem_offers.parquet") -> pd.DataFrame:
    return _load_table(path)


def load_activation_ts(path: str | Path = "data/activation_timeseries.parquet") -> pd.DataFrame:
    return _load_table(path)