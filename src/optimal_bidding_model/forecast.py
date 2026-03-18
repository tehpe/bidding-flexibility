import pandas as pd

#%% Shared helpers

def _with_datetime(df):
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    return x

def _get_bid_prices(config, key):
    return [config[key][i] for i in range(1, config["n_bids"] + 1)]

#%% RCM: estimating acceptance probabilities

def get_price_window(df, product, forecast_date, window_days):
    x = _with_datetime(df)
    forecast_date = pd.to_datetime(forecast_date)

    prices = x.loc[
        (x["product"] == product)
        & x["date"].between(
            forecast_date - pd.Timedelta(days=window_days),
            forecast_date - pd.Timedelta(days=1),
        ),
        "marginal_price",
    ].sort_index()

    return prices

def empirical_cdf(window, p):
    return sum(x <= p for x in window) / len(window)

def get_cdf_values(window, bid_prices):
    return [empirical_cdf(window, p) for p in bid_prices]

def cdf_to_accept_probabilities(cdf_values):
    q = [cdf_values[0]]
    q.extend(cdf_values[i] - cdf_values[i - 1] for i in range(1, len(cdf_values)))
    q.append(1 - cdf_values[-1])
    return q

def q_k(df, config, forecast_date):
    window_days = config["forecast"]["window_days"]
    bid_prices = _get_bid_prices(config, "bid_price_rc_data")
    accept_prob_data = {}

    for product in config["products"]:
        window = get_price_window(df, product, forecast_date, window_days)
        cdf_values = get_cdf_values(window, bid_prices)
        q_values = cdf_to_accept_probabilities(cdf_values)

        if abs(sum(q_values) - 1.0) > 1e-9:
            raise ValueError(f"{product}: acceptance probabilities do not sum to 1")

        accept_prob_data[product] = dict(enumerate(q_values))

    return accept_prob_data


#%% REM estimating activation durations

def L_k(activation_df, rem_offers_df, d, product, p):
    d = pd.to_datetime(d)

    offers_day = rem_offers_df.loc[
        (rem_offers_df["date"] == d) & (rem_offers_df["product"] == product)
    ]
    threshold = offers_day.loc[offers_day["energy_price"] <= p, "allocated_capacity"].sum() # Psi_k(p)

    activated_day = activation_df.loc[
        (activation_df["date"] == d) & (activation_df["product"] == product),
        "activated_mw"] # S_t
    
    return int((activated_day >= threshold).sum())

def summary_stat(values, method, q=None): # method for different summary statistics
    s = pd.Series(values)
    if method == "quantile":
        if q is None:
            raise ValueError("q required for quantile")
        return s.quantile(q)
    if method == "mean":
        return s.mean()
    raise ValueError(f"Undefined method: {method}")

def alpha_k_all_prices( activation_df: pd.DataFrame, rem_offers_df: pd.DataFrame, forecast_date,
    product: str, bid_prices: dict, window_days: int, method: str, q=None) -> dict:

    # date range
    forecast_date = pd.to_datetime(forecast_date)
    start = forecast_date - pd.Timedelta(days=window_days)
    end = forecast_date - pd.Timedelta(days=1)

    act = activation_df.loc[
        activation_df["product"].eq(product)
        & activation_df["date"].between(start, end),
        ["date", "activated_mw"],
    ]

    # prepare data slices
    offers = rem_offers_df.loc[
        rem_offers_df["product"].eq(product)
        & rem_offers_df["date"].between(start, end),
        ["date", "energy_price", "allocated_capacity"],
    ]

    # activation by day (day-based lookup dictionary)
    act_by_day = {
        d: (g["activated_mw"].abs().to_numpy() if product.startswith("NEG_")
            else g["activated_mw"].to_numpy())
        for d, g in act.groupby("date", sort=True)
    }
    
    # offers by day: sorted prices + cumulative capacity (=daily supply curve)
    offers_by_day = {}
    for d, g in offers.groupby("date", sort=True):
        g = g.sort_values("energy_price")
        offers_by_day[d] = (
            g["energy_price"].to_numpy(),
            g["allocated_capacity"].cumsum().to_numpy(),
        )

    out = {}

    for i, p in bid_prices.items():
        daily_durations = []

        for d, activated in act_by_day.items():
            prices, cumcap = offers_by_day.get(d, (None, None))

            if prices is None:
                threshold = 0.0
            else:
                idx = prices.searchsorted(p, side="right") - 1
                threshold = 0.0 if idx < 0 else float(cumcap[idx])

            daily_durations.append(int((activated >= threshold).sum()))

        out[i] = summary_stat(daily_durations, method, q)

    return out



