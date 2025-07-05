from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import List
from agents.utils import utiltiy
import io
import re
from rich import print

OperationPrompts = utiltiy.load_prompt_config(
    r"C:\Users\nigam\Documents\AutoMMM\agents\prompts\data_analysis_agent.yaml",
    "OperationPrompts",
)

llm = ChatOllama(model="llama3.1")


class ColumnCategories(BaseModel):
    date_col: str = Field(description="The date column from the list of columns")
    product_col: str = Field(
        description="The product description column from list of columns"
    )
    sales_cols: List[str] = Field(
        description="The sales related columns from list of columns, like sales, price, sold units"
    )
    oos_col: str = Field(description="The oos column from list of columns")
    media_spends_cols: List[str] = Field(
        description="The spends or costs columns from list of columns"
    )
    media_clicks_cols: List[str] = Field(
        description="The clicks or impression columns from list of columns"
    )
    control_variables: List[str] = Field(
        description="remaining Other columns which affect the sales"
    )

class ColumnCategoriesResponse(BaseModel):
    column_categories : ColumnCategories
    thought_process : str
    all_columns: List[str]


class DataOperations:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_context = (
            open(r"C:\Users\nigam\Documents\AutoMMM\memory\column_desc.txt", "r")
            .read()
            .strip()
            .replace("  ", " ")
            .replace("  ", " ")
        )
        self.data_context = (
            open(r"C:\Users\nigam\Documents\AutoMMM\memory\data_desc.txt", "r")
            .read()
            .strip()
            .replace("  ", " ")
            .replace("  ", " ")
        )
        self.domain_context = (
            open(r"C:\Users\nigam\Documents\AutoMMM\memory\domain_desc.txt", "r")
            .read()
            .strip()
            .replace("  ", " ")
            .replace("  ", " ")
        )

    def categorize_columns(self, history : list):
        llm_structured = llm.with_structured_output(ColumnCategoriesResponse)
        messages = [{'role':'system', 'content': OperationPrompts["ColumnCatogerizer"]},        
                    {'role':'user', 'content': f"input - {list(self.df.columns)}"}]       
        if len(history)>0:
            messages.extend(history)
        # print("test=============>")
        # print("messages: ", messages)
        response = llm_structured.invoke(messages)
        
        # print("test=============>")
        # print("messages: ", messages)
        # print("\n\nresponse: ",response)
        # print("================")
        return dict(response.column_categories)

    def distinctProductIdentification(self, column_categories: dict):
        unique_products = (
            self.df[column_categories["product_col"]].dropna().unique().tolist()
        )
        return unique_products

    def SchemaValidation(self,column_categories: dict):
        all_categorized = []
        for cols in column_categories.values():
            if isinstance(cols, list):
                all_categorized += cols
            else:
                all_categorized.append(cols)
        if set(all_categorized).intersection(set(self.df.columns)):
            column_missed = set(all_categorized).intersection(set(self.df.columns))

        # Type checks
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        df_info_str = buffer.getvalue()

        cleaned_lines = []
        for line in df_info_str.splitlines():
            cleaned_line = re.sub(r"\s{2,}", "|", line.strip())
            cleaned_lines.append(cleaned_line)
        cleaned_df_info_str = "\n".join(cleaned_lines)

        typecheckPrompt = OperationPrompts["TypeChecks"].format(
            domain_context=self.domain_context,
            column_context=self.column_context,
            column_info=cleaned_df_info_str
        )
        typeChecks = llm.invoke(typecheckPrompt)

        # Check for duplicate (date, product) pairs
        duplicate_keys = self.df[
            self.df.duplicated(
                subset=[
                    column_categories["date_col"],
                    column_categories["product_col"],
                ]
            )
        ][[column_categories["date_col"], column_categories["product_col"]]]

        return column_missed, typeChecks, duplicate_keys

    def MissingValueCheck(self, column_categories: dict, product_id:str):
        prod_df = self.df[self.df[column_categories["product_col"]]==product_id]

        all_categorized = []
        for cols in column_categories.values():
            if isinstance(cols, list):
                all_categorized += cols
            else:
                all_categorized.append(cols)
        
        missing_summary = (prod_df[all_categorized].isnull().mean()).to_dict() # type: ignore
        return missing_summary
    
    def OutlierDetection(self, column_categories: dict, product_id):

        prod_df = self.df[self.df[column_categories["product_col"]]==product_id]
        method = "iqr"
        iqr_multiplier = 1.5
        z_thresh = 3.0

        report = {"actions_suggested": [], "outlier_summary": {}}

        numeric_cols = (
            column_categories["sales_cols"]
            + column_categories["media_spends_cols"]
            + column_categories["media_clicks_cols"]
        )
        outlier_records = []

        for col in numeric_cols:
            if col not in prod_df.columns:
                continue
            for product, group in prod_df.groupby(column_categories["product_col"]):
                series = group[col].dropna()
                if series.empty:
                    continue

                if method == "iqr":
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR

                    outliers = group[
                        (group[col] < lower_bound) | (group[col] > upper_bound)
                    ]
                else:  # z-score method
                    mean = series.mean()
                    std = series.std()
                    z_scores = (group[col] - mean) / std
                    outliers = group[np.abs(z_scores) > z_thresh]

                if not outliers.empty:
                    for idx, val in outliers[col].items(): # type: ignore
                        outlier_records.append(
                            {
                                "product": product,
                                "column": col,
                                "index": idx,
                                "value": val,
                            }
                        )
        return pd.DataFrame(outlier_records).to_dict()

    def DateSeriesConsistency(self, column_categories:dict):
        report = {"date_issues": []}
        df_ = self.df.sort_values(by=[column_categories["product_col"], column_categories["date_col"]])
        # Full weekly index per product
        all_prods = df_[column_categories["product_col"]].unique()

        for prod in all_prods:
            prod_df = df_[df_[column_categories["product_col"]] == prod].copy()
            idx = pd.date_range(
                prod_df[column_categories["date_col"]].min(),
                prod_df[column_categories["date_col"]].max(),
                freq="W",
            )
            idx.name = column_categories["date_col"]
            full_idx = pd.MultiIndex.from_product(
                [[prod], idx], names=[column_categories["product_col"], column_categories["date_col"]]
            )
            prod_df = prod_df.set_index([column_categories["product_col"], column_categories["date_col"]])
            # prod_df = prod_df.reindex(full_idx)
            missing_weeks = prod_df.index.get_level_values(column_categories["date_col"])[
                prod_df.isnull().any(axis=1)
            ].to_list()
            if missing_weeks:
                report["date_issues"].append(
                    f"[Observation]: Missing weeks for product {prod}: {missing_weeks}"
                )

        return report

    def BusinessSanitycheck(self, column_categories:dict):
        # Negative values in sales and media
        report = {"sanity_issues": []}
        for col in (
            column_categories["sales_cols"]
            + column_categories["media_spends_cols"]
            + column_categories["media_clicks_cols"]
        ):
            if col in self.df.columns:
                negs = self.df[self.df[col] < 0]
                if not negs.empty:
                    report["sanity_issues"].append(
                        f"Negative values in {col}, count: {len(negs)}"
                    )

        # Zero or negative prices when units_sold > 0
        if "price" in column_categories["sales_cols"] and "units_sold" in column_categories["sales_cols"]:
            bad_price = self.df[
                (self.df["units_sold"] > 0) & ((self.df["price"] <= 0) | self.df["price"].isna())
            ]
            if not bad_price.empty:
                report["sanity_issues"].append(
                    f"Zero or negative prices when units_sold > 0, count: {len(bad_price)}"
                )

        # Price consistency: price ≈ revenue / units_sold
        if all(x in self.df.columns for x in ["price", "revenue", "units_sold"]):
            df_nonzero = self.df[
                (self.df["units_sold"] > 0) & self.df["revenue"].notna() & self.df["price"].notna()
            ]
            price_calc = df_nonzero["revenue"] / df_nonzero["units_sold"]
            inconsistent = df_nonzero[
                np.abs(df_nonzero["price"] - price_calc) > 0.05 * price_calc 
            ]
            if len(inconsistent) != 0:
                report["sanity_issues"].append(
                    f"Price inconsistent with revenue/units_sold in {len(inconsistent)} rows"
                )

        return report

