from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import List, Literal
# from langchain_experimental.tools import PythonAstREPLTool

from contextlib import redirect_stdout
from io import StringIO
import pandas as pd
import io
import numpy as np
import re

from agent_patterns.structured_response import ColumnCategories
from utils.memory_handler import DataStore


class sum_tool(BaseTool):
    name: str = "addition"
    description: str = """
    eg: add_tool : 5 + 3
    returns the sum of two numbers
    """

    def _run(self, a: int, b: int):
        return a + b
# Misc
@tool
def execute_python_code_on_df(code: str, df_required: bool):
    """
    Executes Python code and returns add prints to return the output.
    If the query is related to data, provide 'df' in the execution environment,
    else run code normally without 'df'.
    """
    df = DataStore.get_df("master_data")
    try:
        io_buffer = StringIO()

        env = {}
        if df_required:
            env["df"] = df

        with redirect_stdout(io_buffer):
            exec(code, env)
        return io_buffer.getvalue()
    except Exception as e:
        return f"Error: {type(e).__name__} - {str(e)}"

# Data Level
@tool
def generate_validation_summary(df_required: bool = True):
    "Tool to generate a summary report of dataset shape and columns"

    df = DataStore.get_df("master_data")    
    return f"Shape: {df.shape}, Columns: {list(df.columns)}"

@tool
def data_describe(df_required: bool = True):
    "checking data types of columns"
    df = DataStore.get_df("master_data")
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()

    cleaned_lines = []
    for line in df_info_str.splitlines():
        cleaned_line = re.sub(r"\s{2,}", "|", line.strip())
        cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)

@tool
def validate_column_name_format(df_required: bool = True):
    "Tool to validate column name formats"
    df = DataStore.get_df("master_data")
    bad_cols = [col for col in df.columns if ' ' in col or not col.isidentifier()]
    return f"Bad column names: {bad_cols}" if bad_cols else "All column names are clean."

@tool
def validate_date_format(df_required: bool = True):
    "Tool to verify date column format correctness"
    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))
    try:
        pd.to_datetime(df[column_config['date_col']])
        return "Date column is in valid format."
    except Exception as e:
        return f"Invalid date format: {e}"

@tool
def validate_data_types(df_required: bool = True):
    "Tool to report data types of all columns"
    df = DataStore.get_df("master_data")
    return f"Data types: {df.dtypes.to_dict()}"

@tool
def duplicate_checker(df_required: bool = True):
    """Check for duplicate (date, product) pairs"""
    
    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))
    
    duplicate_keys = df[
        df.duplicated(
            subset=[
                column_config["date_col"],
                column_config["product_col"],
            ]
        )
    ][[column_config["date_col"], column_config["product_col"]]]
    return duplicate_keys.to_json(orient="records")

@tool
def validate_time_granularity(df_required: bool = True):
    "Tool to analyze the time granularity of data"
    
    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))
    
    df[column_config['date_col']] = pd.to_datetime(df[column_config['date_col']])
    freq = df[column_config['date_col']].diff().dropna().value_counts()
    return f"Time frequency distribution:\n{freq}"

@tool
def raise_validation_warnings(df_required: bool = True):
    "Tool to raise warnings on critical data anomalies"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))
    
    warnings = []
    if df[column_config['revenue_col']].min() < 0:
        warnings.append("Negative revenue values found.")
    if df[column_config['units_sold_col']].max() > 1e6:
        warnings.append("Unusually high units sold.")
    return "\n".join(warnings) if warnings else "No critical warnings."

# Product Level
@tool
def validate_missing_dates(product_id: str, df_required: bool = True):
    "Tool to identify gaps in the date sequence"
    
    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))
    
    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    filtered_df[column_config['date_col']] = pd.to_datetime(filtered_df[column_config['date_col']])
    full_range = pd.date_range(start=filtered_df[column_config['date_col']].min(), end=filtered_df[column_config['date_col']].max(), freq='W')
    missing = set(full_range) - set(filtered_df[column_config['date_col']])
    
    return f"Missing Dates for {product_id} : {sorted(missing)}" if missing else f"Missing Dates for {product_id} : No missing dates." 

@tool
def validate_nulls(product_id: str, df_required: bool = True):
    "Tool to find missing (null) values in data"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    nulls = filtered_df.isnull().sum()
    return f"Null values for {product_id} : {nulls[nulls > 0]}" if nulls.any() else f"Null values for {product_id} : No null values." 

@tool
def validate_duplicate_rows(product_id: str, df_required: bool = True):
    "Tool to identify duplicate rows in the data"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    duplicates = filtered_df[filtered_df.duplicated(keep=False)]
    if duplicates.empty:
        return "No duplicate rows."    
    return duplicates.to_json(orient="records")

@tool
def validate_unique_time_product(product_id: str, df_required: bool = True):
    "Tool to ensure uniqueness of (date, product_id) pairs"    

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    dupes = filtered_df.duplicated(subset=[column_config['date_col'], column_config['product_col']])
    dupes_sum = dupes.sum()
    return f"duplicate (date, product_id) pairs for {product_id} : {dupes_sum}" if dupes_sum else f"duplicate (date, product_id) pairs for {product_id} : All (date, product_id) pairs are unique."

@tool
def validate_price_consistency(product_id: str, df_required: bool = True):
    "Tool to validate price calculation consistency"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()

    with np.errstate(divide='ignore', invalid='ignore'):
        price = filtered_df[column_config['revenue_col']] / filtered_df[column_config['units_sold_col']]
        bad_rows = (price < 0) | (np.isinf(price)) | (price.isnull())
        return f"Inconsistent price for {product_id}: {bad_rows.sum()} rows"

@tool
def validate_outliers(product_id: str, df_required: bool = True):
    "Tool to detect outliers using the IQR method"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    outliers = {}
    for col in filtered_df.select_dtypes(include=np.number).columns:
        q1 = filtered_df[col].quantile(0.25)
        q3 = filtered_df[col].quantile(0.75)
        iqr = q3 - q1
        out = filtered_df[(filtered_df[col] < q1 - 1.5 * iqr) | (filtered_df[col] > q3 + 1.5 * iqr)]
        outliers[col] = len(out)
    return f"Outliers detected for {product_id} :{outliers}"

@tool
def validate_minimum_duration(product_id: str, df_required: bool = True):
    "Tool to confirm minimum historical data duration"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    filtered_df[column_config['date_col']] = pd.to_datetime(filtered_df[column_config['date_col']])
    duration = (filtered_df[column_config['date_col']].max() - filtered_df[column_config['date_col']].min()).days
    return f"Duration for {product_id} : {duration} days"

@tool
def validate_media_spend_coverage(product_id: str, df_required: bool = True):
    "Tool to check completeness of media spend data"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    media_cols = [col for col in filtered_df.columns if 'spend' in col]
    missing_data = filtered_df[media_cols].isnull().sum().to_dict()
    return f"Media spend nulls for {product_id} :{missing_data}"

@tool
def validate_oos_sales(product_id: str, df_required: bool = True):
    "Tool to verify out-of-stock periods align with sales drops"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    high_oos_low_sales = filtered_df[(filtered_df[column_config['oos_col']] > 0.5) & (filtered_df[column_config['units_sold_col']] == 0)]
    return f"OOS-Sales mismatch HIGH-OOS LOW-SALES count for {product_id} : {len(high_oos_low_sales)}"

@tool
def validate_seasonality(product_id: str, df_required: bool = True):
    "Tool to assess seasonality effects in sales data"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    filtered_df[column_config['date_col']] = pd.to_datetime(filtered_df[column_config['date_col']])
    filtered_df['week'] = filtered_df[column_config['date_col']].dt.isocalendar().week
    grouped = filtered_df.groupby('week')['units_sold_col'].mean()
    return f"Avg sales by week for {product_id} : \n{grouped.to_dict()}"

@tool
def validate_media_sales_sync(product_id: str, df_required: bool = True):
    "Tool to analyze correlation between media spend and sales"

    df = DataStore.get_df("master_data")
    column_config_df = DataStore.get_df("column_config_df")
    column_config = dict(zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"]))

    filtered_df = df[df[column_config['product_col']] == product_id].copy()
    media_cols = [col for col in filtered_df.columns if 'spend' in col]
    corrs = {col: filtered_df[col].corr(filtered_df[column_config['units_sold_col']]) for col in media_cols}
    return f"Correlation with sales for {product_id} : {corrs}"








