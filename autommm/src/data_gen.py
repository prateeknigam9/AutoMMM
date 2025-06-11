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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

seed: int = sum(map(ord, "mmm"))
rng: np.random.Generator = np.random.default_rng(seed=seed)

data_gen_params = {
    'products' : {
        'premium': 80,
        'mid_tier': 30,
        'low_tier': 20
    },
    'min_date': pd.to_datetime("2023-01-01"), # YYYY-MM-DD
    'max_date': pd.to_datetime("2025-01-01"), # YYYY-MM-DD
    'kpis' :{
            'branded': 0.3,
            'nonbranded': 0.2,
            'insta': 2,
            'fb': 3
    },
    'events' : [ # YYYY-MM-DD
        "2024-05-13", # launch of a particular product
        "2025-09-14"  # Prime day
        ], 
    'kpi_coefs' :{
            'branded': 3.0,
            'nonbranded': 2.0,
            'insta': 1.2,
            'fb': 0.8
    },
}

def gen_rand_media_scaled(n_rows :int):
    return rng.uniform(low=0.0, high=1.0, size=n_rows)

def apply_adstock(df_col: pd.Series, alpha : float ):
    return (
        geometric_adstock(x=df_col.to_numpy(), alpha=alpha, l_max=8, normalize=True)
        .eval()
      .flatten()
    )

def apply_saturation(df_col: pd.Series, lambda_: float):
    return logistic_saturation(
        x=df_col.to_numpy(), lam=lambda_
    ).eval()

# for indiv sku
def data_gen_for_one_sku(config: dict):
    # date outline
    df = pd.DataFrame(
        data={"date_week": pd.date_range(start=config['min_date'], end=config['max_date'], freq="W-SAT")}
    ).assign(
        year=lambda x: x["date_week"].dt.year,
        month=lambda x: x["date_week"].dt.month,
        dayofyear=lambda x: x["date_week"].dt.dayofyear,
    )

    n = df.shape[0]

    # Media Kpis
    random_drop = [0, 1/2, 1/3, 1/4]
    random_alpha = [0, 0.4, 0.2, 0.3]
    random_lambda = [1, 2, 3, 4, 5]
    for kpi,cpc in config['kpis'].items():
        x = gen_rand_media_scaled(n_rows = n)
        prop = random.choice(random_drop)
        df[kpi+'_clicks'] = np.where(x > 0.9, x, x * prop)
        df[kpi+'_spends'] = df[kpi+'_clicks'] * cpc

        df[kpi+'_clicks_adstock'] = apply_adstock(df_col = df[kpi+'_clicks'],
                                                   alpha = random.choice(random_alpha))
        df[kpi+'_clicks_adstock_saturated'] = apply_saturation(df_col = df[kpi+'_clicks_adstock'],
                                                   lambda_= random.choice(random_lambda))
    
    # Trend and seasonal component
    df["trend"] = (np.linspace(start=0.0, stop=50, num=n) + 10) ** (1 / 4) - 1

    df["cs"] = -np.sin(2 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["cc"] = np.cos(1 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["seasonality"] = 0.5 * (df["cs"] + df["cc"])

    # Control Variables
    df["event"] = df["date_week"].isin(config["events"]).astype(float)

    # target column - unit sales
    df["intercept"] = random.choice([2.0, 3.0, 5.0])
    df["epsilon"] = rng.normal(loc=0.0, scale=0.25, size=n)
    amplitude = 1

    units_sold = amplitude * (
        df["intercept"]
        + df["trend"]
        + df["seasonality"]
        + 1.5 * df["event"]
        + df["epsilon"]
    )
    for kpi,kpi_coef in config['kpi_coefs'].items():
        units_sold += kpi_coef * df[kpi+'_clicks_adstock_saturated']

    df["units_sold"] = units_sold

    return df

# TODO : price, OOS
def generate_synthetic_data(config: dict) -> pd.DataFrame:
    data_frames = []
    for product_id, avg_price in tqdm(config['products'].items(), desc="Generating data"):
        df = data_gen_for_one_sku(config)
        df["product_id"] = product_id
        df["avg_price"] = avg_price
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


def main():
    synthetic_data = generate_synthetic_data(config=data_gen_params)
    synthetic_data.to_excel("autommm/data/synthetic_data.xlsx", index=False)
    print("Synthetic data saved to 'data/synthetic_data.xlsx'.")

if __name__ == "__main__":
    main()