"""
Prepare model input data from raw external market files.
See README.md for download sources.
"""
from pathlib import Path
import pandas as pd
from optimal_bidding_model.data import load_config


ROOT = Path(__file__).resolve().parents[1]
# ROOT = Path("/Users/tehpe/Desktop/GitProjects/optimal bidding model")
CONFIG_FILE = ROOT / "src" / "optimal_bidding_model" / "config.yaml"
RAW_DIR = ROOT / "data" / "tmp-raw"
OUT_DIR = ROOT / "data"


def filter_period_and_products(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    start = pd.to_datetime(config["period"]["start"])
    end = pd.to_datetime(config["period"]["end"])
    products = config["products"]

    return df[
        df["date"].between(start, end) & df["product"].isin(products)
    ].sort_values(["date", "product"])


def process_marginal_prices(config: dict) -> None:
    df = pd.read_excel(
        RAW_DIR / "RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2021.xlsx",
        parse_dates=["DATE_TO"],
    ).rename(
        columns={
            "DATE_TO": "date",
            "PRODUCT": "product",
            "GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]": "marginal_price"
        }
    )[["date", "product", "marginal_price"]]

    df = filter_period_and_products(df, config)

    # save
    df.to_parquet(OUT_DIR / "marginal_prices.parquet", index=False)
    print("Saved marginal_prices.parquet")



def process_rem_offers(config: dict) -> None:
    df = pd.concat(
        pd.read_excel(
            RAW_DIR / "RESULT_LIST_ANONYM_ENERGY_MARKET_aFRR_2021.xlsx",
            sheet_name=None,
        ).values(),
        ignore_index=True
    ).rename(
        columns={
            "DELIVERY_DATE": "date",
            "PRODUCT": "product",
            "ENERGY_PRICE_[EUR/MWh]": "energy_price",
            "ALLOCATED_CAPACITY_[MW]": "allocated_capacity"
        }
    )

    df = df[df["COUNTRY"].astype(str).str.strip().str.upper() == "DE"].copy() # Germany-only
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce") 
    df = df[["date", "product", "energy_price", "allocated_capacity"]]

    df = filter_period_and_products(df, config)

    # save
    df.to_parquet(OUT_DIR / "rem_offers.parquet", index=False)
    print("Saved rem_offers.parquet")


def process_activation_times(config: dict) -> None:
    df = pd.read_csv(
        RAW_DIR / "SECONDS_BASE_AFRR_TARGET_VALUES_2021.csv",
        sep=";",
        dtype={"DATE": str, "TIME": str}
    ).rename(columns={"GERMANY_aFRR_SETPOINT_[MW]": "activated_mw"})

    df["activated_mw"] = pd.to_numeric(
        df["activated_mw"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(
        df["DATE"] + " " + df["TIME"],
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce"
    )

    hour = df["timestamp"].dt.hour
    block_start = (hour // 4) * 4

    df["date"] = df["timestamp"].dt.normalize()
    df["product"] = (
        df["activated_mw"].ge(0).map({True: "POS", False: "NEG"})
        + "_"
        + block_start.astype(str).str.zfill(2)
        + "_"
        + (block_start + 4).astype(str).str.zfill(2)
    )
    df["second"] = (
        (hour - block_start) * 3600
        + df["timestamp"].dt.minute * 60
        + df["timestamp"].dt.second
    )

    df = df[["date", "product", "second", "activated_mw"]].dropna()
    df = filter_period_and_products(df, config)

    # save 
    df.to_parquet(OUT_DIR / "activation_timeseries.parquet", index=False)
    print("Saved activation_timeseries.parquet")


def main() -> None:
    config = load_config(CONFIG_FILE)
    OUT_DIR.mkdir(exist_ok=True)

    process_marginal_prices(config)
    process_rem_offers(config)
    process_activation_times(config)


if __name__ == "__main__":
    main()