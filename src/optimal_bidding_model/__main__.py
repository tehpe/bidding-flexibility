import pandas as pd
import pyomo.environ as pyo

from optimal_bidding_model.data import *
from optimal_bidding_model.model import build_model
from optimal_bidding_model.forecast import (q_k, alpha_k)

def main():

    config = load_config("config.yaml")
    config["n_bids"]  = len(config["bid_price_rc_data"])
    config["m_bids"]  = len(config["bid_price_re_data"])

    # data
    marginal_prices = load_marginal_prices("data/marginal_prices.csv")
    rem_offers      = load_rem_offers("data/rem_offers.csv")
    activation_ts   = load_activation_ts("data/activation_timeseries.csv")
    forecast_date = marginal_prices["date"].max() + pd.Timedelta(days=1)


    # Forecast input parameters
    config["accept_prob_data"] = q_k(
        marginal_prices,
        config,
        forecast_date
    )

    config["activation_duration_data"] = {k: {} for k in config["products"]} # init

    for k in config["products"]: 
        for i in config["bid_price_re_data"]:
            bid_price = config["bid_price_re_data"][i]

            config["activation_duration_data"][k][i]= alpha_k( #assign
                activation_ts, rem_offers, forecast_date, 
                product=k, p=bid_price, q=None, 
                window_days=config["forecast"]["window_days"], 
                method=config["forecast"]["alpha_method"])

    # build model
    model = build_model(config)

    # solve
    solver = pyo.SolverFactory(config["solver"]["name"])
    result = solver.solve(model)

    # extract results
    if (
        result.solver.status == pyo.SolverStatus.ok
        and result.solver.termination_condition == pyo.TerminationCondition.optimal
    ):
        print("\nRESULTS:")
        print(f"opt obj value: {pyo.value(model.obj)}")

        print("\nReserve capacity market:")
        for k in model.K:
            print(f"{k}:")
            for i in model.N:
                print(f"  m_rc[{k},{i}] = {pyo.value(model.m_rc[k, i])}")
            print(f"  f_rc_k[{k}] = {pyo.value(model.f_rc_k[k])}")
            print()

        print("\nReserve energy market:")
        for k in model.K:
            print(f"{k}:")
            for i in model.M:
                print(f"  m_re[{k},{i}] = {pyo.value(model.m_re[k, i])}")
            print(f"  f_re_k[{k}] = {pyo.value(model.f_re_k[k])}")
            print()
    else:
        print("Solver did not find an optimal solution.")

if __name__ == "__main__":
    main()