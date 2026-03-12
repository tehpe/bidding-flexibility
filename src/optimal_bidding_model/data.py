from pathlib import Path
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_csv(path):
    return pd.read_csv(path, parse_dates=["date"])


def load_marginal_prices(path: str | Path = "data/marginal_prices.csv") -> pd.DataFrame:
    return _load_csv(path)


def load_rem_offers(path: str | Path = "data/rem_offers.csv") -> pd.DataFrame:
    return _load_csv(path)


def load_activation_ts(path: str | Path = "data/activation_timeseries.csv") -> pd.DataFrame:
    return _load_csv(path)