import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
import random
from tqdm import tqdm

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
            "branded": {"premium": 6, "mid_tier": 5, "low_tier": 2},
            "nonbranded": {"premium": 7, "mid_tier": 9, "low_tier": 4}
            # "oos": {"premium": 7, "mid_tier": 9, "low_tier": 4},
        },
        "brand_level": {"insta": 1.2, "fb": 0.8},
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

    # Trend and seasonal component
    df["trend"] = (np.linspace(start=0.0, stop=50, num=n) + 10) ** (1 / 4) - 1

    df["cs"] = -np.sin(2 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["cc"] = np.cos(1 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["seasonality"] = 0.5 * (df["cs"] + df["cc"])

    # Control Variables
    df["event"] = df["date_week"].isin(config["events"]).astype(float)

    # individual product property
    df["intercept"] = random.choice([2.0, 3.0, 5.0])
    df["epsilon"] = rng.normal(loc=0.0, scale=0.25, size=n)

    return df


def brand_level_data(config:dict, n_rows: int):
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
            df_col=brand_df[kpi + "_clicks_adstock"], lambda_=random.choice(random_lambda)
        )
    return brand_df



# TODO : price, OOS
def generate_synthetic_data(config: dict, n_rows: int) -> pd.DataFrame:
    brand_data = brand_level_data(config, n_rows)
    data_frames = []

    for product_id, avg_price in tqdm(config["products"].items(), desc="Generating data"):
        df = data_gen_for_one_sku(config)
        df["product_id"] = product_id
        df["avg_price"] = avg_price

        complete_data = pd.concat([df, brand_data], axis=1)

        # Start units_sold calculation
        units_sold = (
            complete_data["intercept"]
            + complete_data["trend"]
            + complete_data["seasonality"]
            + 1.5 * complete_data["event"]
            + complete_data["epsilon"]
        )

        # Apply product-level KPIs
        for kpi, tiers in config["kpi_coefs"]["product_level"].items():
            if product_id in tiers:
                coef = tiers[product_id]
                units_sold += coef * complete_data[f"{kpi}_clicks_adstock_saturated"]

        # Apply brand-level KPIs
        for kpi, coef in config["kpi_coefs"]["brand_level"].items():
            units_sold += coef * complete_data[f"{kpi}_clicks_adstock_saturated"]

        complete_data["units_sold"] = units_sold
        complete_data["revenue"] = complete_data["units_sold"] * complete_data["avg_price"]

        data_frames.append(complete_data)

    return pd.concat(data_frames, ignore_index=True)


def main():
    synthetic_data = generate_synthetic_data(config=data_gen_params,n_rows=104)
    synthetic_data.to_excel("autommm/data/synthetic_data.xlsx", index=False)
    print("Synthetic data saved to 'data/synthetic_data.xlsx'.")


if __name__ == "__main__":
    main()
