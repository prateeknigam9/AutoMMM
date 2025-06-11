import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os

# ---- TEMP FIX FOR MODULE IMPORTS ----
# Adds project root (3 levels up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

seed: int = sum(map(ord, "mmm"))
rng: np.random.Generator = np.random.default_rng(seed=seed)

data_gen_params = {
    # products and avg_prices
    "products": {"premium": 80, "mid_tier": 30, "low_tier": 20},
    # Min date  & max date YYYY-MM-DD
    "min_date": pd.to_datetime("2023-01-01"),
    "max_date": pd.to_datetime("2025-01-01"),
    # KPI and CPC
    "kpis": {
        "product_level": {"branded": 0.3, "nonbranded": 0.2},
        "brand_level": {"insta": 2, "fb": 3},
    },
    # EVENTS YYYY-MM-DD
    "events": [
        "2024-05-13",  # launch of a particular product
        "2025-09-14",  # Prime day
    ],
    # Coefficients for every kpi for every product
    "kpi_coefs": {
        "product_level": {
            "branded": {"premium": 15, "mid_tier": 10, "low_tier": 8},
            "nonbranded": {"premium": 12, "mid_tier": 9, "low_tier": 7},
            "price": {"premium": -0.5, "mid_tier": -0.4, "low_tier": -0.3},  
            "oos": {"premium": -1.2, "mid_tier": -0.8, "low_tier": -0.5},  
        },
        "brand_level": {"insta": 2.5, "fb": 2.0},
    },
}


def gen_rand_media_scaled(n_rows: int):
    return rng.uniform(low=0.0, high=1.0, size=n_rows)


def apply_adstock(df_col: pd.Series, alpha: float):
    return (
        geometric_adstock(x=df_col.to_numpy(), alpha=alpha, l_max=8, normalize=True)
        .eval()
        .flatten()
    )


def apply_saturation(df_col: pd.Series, lambda_: float):
    return logistic_saturation(x=df_col.to_numpy(), lam=lambda_).eval()


def generate_price_list(avg_price: float, n_rows: int, std_dev_factor: float = 0.05):
    std_dev = avg_price * std_dev_factor
    prices = np.random.normal(loc=avg_price, scale=std_dev, size=n_rows)
    prices = np.clip(prices, 0.01, None)  # Avoid zero or negative prices

    price_list = np.round(prices, 2).tolist()
    log_price_list = np.round(np.log(prices), 4).tolist()

    return price_list, log_price_list


def oos_gen(n_rows: int):
    oos_days = np.random.choice([1, 2, 3, 4, 5, 6, 7], size=(n_rows,)) * np.random.choice(
        [0, 1], size=(n_rows,), p=[7.0 / 10, 3.0 / 10]
    )
    return oos_days/7


# for indiv sku
def data_gen_for_one_sku(config: dict):
    # date outline
    df = pd.DataFrame(
        data={
            "date_week": pd.date_range(
                start=config["min_date"], end=config["max_date"], freq="W-SAT"
            )
        }
    ).assign(
        year=lambda x: x["date_week"].dt.year,
        month=lambda x: x["date_week"].dt.month,
        dayofyear=lambda x: x["date_week"].dt.dayofyear,
    )

    n = df.shape[0]

    # product level Media Kpis
    random_drop = [0, 1 / 2, 1 / 3, 1 / 4]
    random_alpha = [0, 0.4, 0.2, 0.3]
    random_lambda = [1, 2, 3, 4, 5]
    for kpi, cpc in config["kpis"]["product_level"].items():
        x = gen_rand_media_scaled(n_rows=n)
        prop = random.choice(random_drop)
        df[kpi + "_clicks"] = np.where(x > 0.9, x, x * prop)
        df[kpi + "_spends"] = df[kpi + "_clicks"] * cpc

        df[kpi + "_clicks_adstock"] = apply_adstock(
            df_col=df[kpi + "_clicks"], alpha=random.choice(random_alpha)
        )
        df[kpi + "_clicks_adstock_saturated"] = apply_saturation(
            df_col=df[kpi + "_clicks_adstock"], lambda_=random.choice(random_lambda)
        )

    df["oos"] = oos_gen(n_rows=n)

    # Trend and seasonal component
    df["trend"] = (np.linspace(start=0.0, stop=50, num=n) + 10) ** (1 / 4) - 1
    # df["trend"] = (np.linspace(start=0.0, stop=50, num=n) + 15) ** 0.5

    df["cs"] = -np.sin(2 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["cc"] = np.cos(1 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["seasonality"] = 0.5 * (df["cs"] + df["cc"])

    # Control Variables
    df["event"] = df["date_week"].isin(config["events"]).astype(float)

    # individual product property
    df["intercept"] = random.choice([15.0, 20.0, 25.0]) 
    df["epsilon"] = rng.normal(loc=0.0, scale=0.25, size=n)

    return df


def brand_level_data(config: dict, n_rows: int):
    brand_df = pd.DataFrame()

    # brand level Media Kpis
    random_drop = [0, 1 / 2, 1 / 3, 1 / 4]
    random_alpha = [0, 0.4, 0.2, 0.3]
    random_lambda = [1, 2, 3, 4, 5]
    for kpi, cpc in config["kpis"]["brand_level"].items():
        x = gen_rand_media_scaled(n_rows=n_rows)
        prop = random.choice(random_drop)
        brand_df[kpi + "_clicks"] = np.where(x > 0.9, x, x * prop)
        brand_df[kpi + "_spends"] = brand_df[kpi + "_clicks"] * cpc

        brand_df[kpi + "_clicks_adstock"] = apply_adstock(
            df_col=brand_df[kpi + "_clicks"], alpha=random.choice(random_alpha)
        )
        brand_df[kpi + "_clicks_adstock_saturated"] = apply_saturation(
            df_col=brand_df[kpi + "_clicks_adstock"],
            lambda_=random.choice(random_lambda),
        )
    return brand_df


# TODO : price, OOS
def generate_synthetic_data(config: dict) -> pd.DataFrame:
    n_rows = pd.date_range(config["min_date"], config["max_date"], freq="W-SAT").shape[
        0
    ]
    brand_data = brand_level_data(config, n_rows)
    data_frames = []

    for product_id, avg_price in tqdm(
        config["products"].items(), desc="Generating data"
    ):
        df = data_gen_for_one_sku(config)
        df["product_id"] = product_id

        # Price
        price_list, log_price_list = generate_price_list(
            avg_price=avg_price, n_rows=n_rows
        )
        df["price"] = price_list
        df["log_price"] = log_price_list

        complete_data = pd.concat([df, brand_data], axis=1)

        # Start units_sold calculation
        units_sold = (
            complete_data["intercept"]
            + complete_data["trend"]
            + complete_data["seasonality"]
            + 3 * complete_data["event"]
            + complete_data["epsilon"]
        )

        # Apply product-level KPIs
        for kpi, tiers in config["kpi_coefs"]["product_level"].items():
            if product_id in tiers:
                coef = tiers[product_id]
                if kpi == "price":
                    units_sold += coef * complete_data["log_price"]
                elif kpi == "oos":
                    units_sold += coef * complete_data["oos"]
                else:
                    units_sold += (
                        coef * complete_data[f"{kpi}_clicks_adstock_saturated"]
                    )

        # Apply brand-level KPIs
        for kpi, coef in config["kpi_coefs"]["brand_level"].items():
            units_sold += coef * complete_data[f"{kpi}_clicks_adstock_saturated"]

        complete_data["units_sold"] = units_sold
        complete_data["revenue"] = complete_data["units_sold"] * complete_data["price"]

        data_frames.append(complete_data)

    return pd.concat(data_frames, ignore_index=True)


def main():
    synthetic_data = generate_synthetic_data(config=data_gen_params)
    synthetic_data.to_excel("autommm/data/synthetic_data.xlsx", index=False)
    print("Synthetic data saved to 'data/synthetic_data.xlsx'.")


if __name__ == "__main__":
    main()
