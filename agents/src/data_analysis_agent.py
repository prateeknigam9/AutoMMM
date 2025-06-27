from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from agents.agent_base import AgentBase
import pandas as pd
import numpy as np
from agents.utils import utiltiy

from langchain.tools import BaseTool

AgentPrompts = utiltiy.load_prompt_config(
    r"C:\Users\nigam\Documents\AutoMMM\agents\prompts\data_analysis_agent.yaml",
    "DataValidator",
)

ToolPrompts = utiltiy.load_prompt_config(
    r"C:\Users\nigam\Documents\AutoMMM\agents\prompts\data_analysis_agent.yaml",
    "ToolPrompts",
)

df = pd.DataFrame()

class DataValidator(AgentBase):
    def __init__(self, client, llm_model, df):
        super().__init__(client)
        self.client = client
        self.llm_model = llm_model
        self.df = df
        # Buildflow.invoke()

    def Think(self, df: pd.DataFrame):
        llm_thoughts = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": AgentPrompts["Think"]["role"]},
                {
                    "role": "user",
                    "content": "Think and understand the context",
                    # "content": AgentPrompts["Think"]["task"] + "Dataset: {df}".strip(),
                },
            ],
        )
        llm_thoughts = llm_thoughts.choices[0].message.content


    def Plan(self):
        llm_thoughts = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": AgentPrompts["Think"]["role"]},
                {
                    "role": "user",
                    "content": "PLan and define actions step by step based on the tools and problem",
                },
            ],
        )
        llm_thoughts = llm_thoughts.choices[0].message.content

    def Action(self):
        llm_thoughts = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": AgentPrompts["Think"]["role"]},
                {
                    "role": "user",
                    "content": "Perform actions in loop based on the available tools",
                },
            ],
        )
        llm_thoughts = llm_thoughts.choices[0].message.content


    def Research(self):
        llm_thoughts = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": AgentPrompts["Think"]["role"]},
                {
                    "role": "user",
                    "content": "research if required",
                },
            ],
        )
        llm_thoughts = llm_thoughts.choices[0].message.content

    def Suggest(self):
        llm_thoughts = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": AgentPrompts["Think"]["role"]},
                {
                    "role": "user",
                    "content": "respond to the user",
                },
            ],
        )
        llm_thoughts = llm_thoughts.choices[0].message.content

    def build_flow(self): 
        ... # Run all of these as workflow
        # self.think
        # self.Plan
        # self.Action
        # self.Research
        # self.action
        # self.response




# THink
# PLan and define actions step by step based on the tools and problem
# Action in loop
# research if required
#  update action
#  Response to user

# llm = ChatOpenAI(temperature=0)

# def analyze_data_tool(input_text: str):
#     return f"Data checked. No missing values in {input_text}"

# tool = Tool(name="AnalyzeData", func=analyze_data_tool, description="Performs data quality check.")

# agent = initialize_agent([tool], llm, agent_type="zero-shot-react-description")

# def run(state):
#     result = agent.run("Check data quality on input dataset")
#     state["last_output"] = result
#     state["next_node"] = "data_insights"
#     return state


# %% TOOLS
####---------------------------------------------####
## TOOLS
####---------------------------------------------####
"""
df
class ColumnCategories(BaseModel):
    date_col : str = Field(description="The date column from the list of columns")
    product_col: str = Field(description="The product description column from list of columns")
    sales_cols: List[str] = Field(description="The sales related columns from list of columns, like sales, price, sold units")
    oos_col : str = Field(description="The oos column from list of columns")
    media_spends_cols : List[str] = Field(description="The spends or costs columns from list of columns")
    media_clicks_cols : List[str] = Field(description="The clicks or impression columns from list of columns")
    control_variables : List[str] = Field(description="remaining Other columns which affect the sales") 
"""

class distinctProduct(BaseTool):
    name: str = "product_identifier"
    description: str = "Tool to identify unique products in the data"

    def _run(self, config: dict):
        unique_products = df[config["product_col"]].dropna().unique().tolist()
        if unique_products is None:
            raise ValueError("Failed to extract unique products from data.")
        return unique_products

    def _arun(self, config: dict):
        raise NotImplementedError("This tool does not support async")


class SchemaValidationTool(BaseTool):
    name: str = "Schema Validation Tool"
    description: str = """
    - Check that all required column groups exist as per `ColumnCategories`
    - Validate data types
    - Date and product must form a unique key per row.
    Output:
        List of missing columns, incorrect types, duplicate key violations.
    """

    def _run(self, config: dict):
        # Define all required columns
        report = {"missing_columns": [], "type_errors": [], "duplicate_keys": []}
        all_required = (
            [config["date_col"]]
            + [config["product_col"]]
            + config["sales_cols"]
            + [config["oos_col"]]
            + config["media_spends_cols"]
            + config["sales_cols"]
            + config["media_clicks_cols"]
            + config["control_variables"]
        )

        # Check for missing columns
        missing = [col for col in all_required if col not in df.columns]
        if missing:
            report["missing_columns"].extend(missing)

        # Type checks for date column
        if config["date_col"] in df.columns:
            try:
                df[config["date_col"]] = pd.to_datetime(df[config["date_col"]])
            except Exception:
                report["type_errors"].append(
                    f"Date column {config['date_col']} not convertible to datetime"
                )

        # Type checks for sales and media columns
        for col in config["sales_cols"] + config["media_spends_cols"]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                report["type_errors"].append(
                    f"Column {col} expected numeric but found {df[col].dtype}"
                )

        # Check for duplicate (date, product) pairs
        if config["date_col"] in df.columns and config["product_col"] in df.columns:
            duplicated = df.duplicated(
                subset=[config["date_col"], config["product_col"]]
            )
            if duplicated.any():
                idx = df[duplicated].index.tolist()
                report["duplicate_keys"].extend(idx)

        return report


class MissingValueCheckTool(BaseTool):
    name: str = "Missing Value Checks"
    description: str = """
        - Profile missingness by column and product.
    Output:
        suggest column-specific remediation:
                Sales/media: interpolate or carry forward.
                Categorical: fill with mode or "unknown".
    """

    def _run(self, config: dict):
        threshold_warn = 0.05
        threshold_fail = 0.20

        report = {
            "missing_data_summary": str,
            "imputation_log": [],
            "actions_suggestions": [],
        }

        all_required = (
            [config["date_col"]]
            + [config["product_col"]]
            + config["sales_cols"]
            + [config["oos_col"]]
            + config["media_spends_cols"]
            + config["sales_cols"]
            + config["media_clicks_cols"]
            + config["control_variables"]
        )


        missing_summary = (df[all_required].isnull().mean()).to_dict() # type: ignore

        report["missing_data_summary"] = (
            "percentage of data missing for columns\n" + str(missing_summary)
        )

        # Imputation log example (simple forward fill or mean for numeric)
        for col, pct_missing in missing_summary.items():
            if pct_missing > 0 and pct_missing <= threshold_warn:
                if (
                    col
                    in config["sales_cols"]
                    + config["media_clicks_cols"]
                    + config["media_spends_cols"]
                ):
                    # Impute numeric with forward fill
                    # df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    report["imputation_log"].append(
                        f"[Suggested]: Forward/backward filled {col}, [reason]: Sales/media: interpolate or carry forward."
                    )
                elif col in config["control_variables"]:
                    mode = df[col].mode()
                    if not mode.empty:
                        # df[col] = df[col].fillna(mode.iloc[0])
                        report["imputation_log"].append(
                            f"[Suggested]: Filled missing {col} with mode: {mode[0]}, [reason]: Categorical: fill with mode"
                        )

            elif pct_missing > threshold_fail:
                report["actions_suggestions"].append(
                    f"[Suggested]: High missingness in {col} ({pct_missing:.2%}), [reason]: consider dropping or further action"
                )

        return report


class OutlierDetectionTool(BaseTool):
    name: str = "Outlier Detection"
    description: str = """
    - Apply IQR-based and z-score methods to `sales_cols` and `media_cols`
    - Detect unusually high or low prices and media spends
    Output:
        Outlier summary report per column.
        List of rows to be modified, capped, or removed with suggestions
    """

    def _run(self, config: dict):
        method = "iqr"
        iqr_multiplier = 1.5
        z_thresh = 3.0

        report = {"actions_suggested": [], "outlier_summary": {}}

        numeric_cols = (
            config["sales_cols"]
            + config["media_spends_cols"]
            + config["media_clicks_cols"]
        )
        outlier_records = []

        for col in numeric_cols:
            if col not in df.columns:
                continue
            for product, group in df.groupby(config["product_col"]):
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
                    # Winsorize (cap) outliers at bounds for IQR method:
                    if method == "iqr":
                        capped_vals = group[col].clip(lower_bound, upper_bound)
                        # df.loc[group.index, col] = capped_vals
                        report["actions_suggested"].append(
                            f"[Suggested]: cap outliers in {col} for product {product}"
                        )

        report["outlier_summary"] = pd.DataFrame(outlier_records).to_dict()
        return report


class DateSeriesConsistencyTool(BaseTool):
    name: str = "Date Series Consistency"
    description: str = """
    - Verify continuity of weekly dates for each product (Duplicate weeks for same product)
    - Flag gaps in the time series
    Output:
        Gantt-style time coverage map.
        Weeks added or removed.
        Duplicate time warnings.
    """

    def _run(self, config: dict):

        report = {"date_issues": []}
        df_ = df.sort_values(by=[config["product_col"], config["date_col"]])
        # Full weekly index per product
        all_prods = df_[config["product_col"]].unique()

        for prod in all_prods:
            prod_df = df_[df_[config["product_col"]] == prod].copy()
            idx = pd.date_range(
                prod_df[config["date_col"]].min(),
                prod_df[config["date_col"]].max(),
                freq="W",
            )
            idx.name = config["date_col"]
            full_idx = pd.MultiIndex.from_product(
                [[prod], idx], names=[config["product_col"], config["date_col"]]
            )
            prod_df = prod_df.set_index([config["product_col"], config["date_col"]])
            prod_df = prod_df.reindex(full_idx)
            missing_weeks = prod_df.index.get_level_values(config["date_col"])[
                prod_df.isnull().any(axis=1)
            ].to_list()
            if missing_weeks:
                report["date_issues"].append(
                    f"[Observation]: Missing weeks for product {prod}: {missing_weeks}"
                )

        return report


class BusinessSanitycheckTool(BaseTool):
    name: str = "Sanity Checks on Business Logic Tool"
    description: str = """
    - `price` should not be zero or negative
    - `revenue` ≈ `price * units_sold`; flag large deviations
    - Media spends/clicks should not be negative
    - Check if `oos_col` (stockouts) align with zero `units_sold`
    Output:
        Violations summary.
        Rows adjusted or to be dropped.
    """

    def _run(self, config: dict):
        # Negative values in sales and media
        report = {"sanity_issues": []}
        for col in (
            config["sales_cols"]
            + config["media_spends_cols"]
            + config["media_clicks_cols"]
        ):
            if col in df.columns:
                negs = df[df[col] < 0]
                if not negs.empty:
                    report["sanity_issues"].append(
                        f"Negative values in {col}, count: {len(negs)}"
                    )

        # Zero or negative prices when units_sold > 0
        if "price" in config["sales_cols"] and "units_sold" in config["sales_cols"]:
            bad_price = df[
                (df["units_sold"] > 0) & ((df["price"] <= 0) | df["price"].isna())
            ]
            if not bad_price.empty:
                report["sanity_issues"].append(
                    f"Zero or negative prices when units_sold > 0, count: {len(bad_price)}"
                )

        # Price consistency: price ≈ revenue / units_sold
        if all(x in df.columns for x in ["price", "revenue", "units_sold"]):
            df_nonzero = df[
                (df["units_sold"] > 0) & df["revenue"].notna() & df["price"].notna()
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
