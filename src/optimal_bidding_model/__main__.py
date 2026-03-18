from pathlib import Path
import time
import pyomo.environ as pyo

from optimal_bidding_model.data import *
from optimal_bidding_model.model import build_model
from optimal_bidding_model.forecast import q_k, alpha_k_all_prices


CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
config = load_config(CONFIG_PATH)


def main() -> None:
    t_start = time.perf_counter()

    config["n_bids"] = len(config["bid_price_rc_data"])
    config["m_bids"] = len(config["bid_price_re_data"])

    forecast_date = config["forecast"]["forecast_date"]

    print("Loading data...")
    marginal_prices = load_marginal_prices("data/marginal_prices.parquet")
    rem_offers = load_rem_offers("data/rem_offers.parquet")
    activation_ts = load_activation_ts("data/activation_timeseries.parquet")

    print("Preparing forecast inputs...")
    config["accept_prob_data"] = q_k(marginal_prices, config, forecast_date)

    config["activation_duration_data"] = {}
    for k in config["products"]:
        config["activation_duration_data"][k] = alpha_k_all_prices(
            activation_ts,
            rem_offers,
            forecast_date=forecast_date,
            product=k,
            bid_prices=config["bid_price_re_data"],
            window_days=config["forecast"]["window_days"],
            method=config["forecast"]["alpha_method"],
            q=config["forecast"].get("alpha_q")
        )

    print("Building model...")
    model = build_model(config)

    print("Starting solve...")
    solver = pyo.SolverFactory(config["solver"]["name"])
    t0 = time.perf_counter()
    result = solver.solve(model, tee=True)
    print(f"Solve time: {time.perf_counter() - t0:.2f}s")

    if (
        result.solver.status == pyo.SolverStatus.ok
        and result.solver.termination_condition == pyo.TerminationCondition.optimal
    ):
        print("\nRESULTS:")
        print(f"opt obj value: {pyo.value(model.obj)}")
        print(f"Total runtime: {time.perf_counter() - t_start:.2f}s")

        tol = 1e-6

        print("\nReserve capacity market:")
        for k in model.K:
            bids = [
                (config["bid_price_rc_data"][i], pyo.value(model.m_rc[k, i]))
                for i in model.N
                if pyo.value(model.m_rc[k, i]) > tol
            ]
            print(f"{k}: {bids}")

        print("\nReserve energy market:")
        for k in model.K:
            bids = [
                (config["bid_price_re_data"][i], pyo.value(model.m_re[k, i]))
                for i in model.M
                if pyo.value(model.m_re[k, i]) > tol
            ]
            print(f"{k}: {bids}")

    else:
        print("Solver did not find an optimal solution.")


if __name__ == "__main__":
    main()