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
    threshold = offers_day.loc[offers_day["ask_price"] <= p, "capacity_mw"].sum() # Psi_k(p)

    activated_day = activation_df.loc[
        (activation_df["date"] == d) & (activation_df["product"] == product),
        "activated_mw"] # S_t
    
    return int((activated_day >= threshold).sum())


def alpha_k(activation_df, rem_offers_df, forecast_date, product, p, window_days, method=None, q=None):
    forecast_date = pd.to_datetime(forecast_date)

    past_days = (
        activation_df.loc[
            (activation_df["product"] == product)
            & activation_df["date"].between(
                forecast_date - pd.Timedelta(days=window_days),
                forecast_date - pd.Timedelta(days=1),
            ),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
    )

    L_values = []
    for d in past_days:
        L_values.append(L_k(activation_df, rem_offers_df, d, product, p))

    L_values = pd.Series(L_values)

    # stat methods
    if method == "mean":
        return L_values.mean()
    if method == "median":
        return L_values.median()
    if method == "quantile":
        if q is None:
            raise ValueError("q must be provided when method='quantile'")
        return L_values.quantile(q)
    raise ValueError(f"Undefined method: {method}")