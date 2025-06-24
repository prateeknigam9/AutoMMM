import pandas as pd
import json
from tabulate import tabulate
from utils.colors import system_message, system_input


def full_dataframe_summary(df, date_col, product_col, price_col, output_json=None):
    summary = {}

    summary["Shape"] = df.shape
    summary["Unique Products"] = df[product_col].nunique()

    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    summary["Missing Values"] = pd.DataFrame(
        {"Count": missing, "Percent": missing_percent}
    ).to_dict(orient="index")

    unique_info = {}
    for col in df.columns:
        unique_vals = df[col].nunique(dropna=False)
        top_value = df[col].mode(dropna=False)
        unique_info[col] = {
            "Unique Values": int(unique_vals),
            "Most Frequent": top_value.iloc[0] if not top_value.empty else None,
        }
    summary["Unique and Most Frequent"] = unique_info

    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            date_series_df = df[df[product_col] == df[product_col].unique()[0]]

            date_series = date_series_df[date_col].dropna().sort_values()
            if not date_series.empty:
                inferred_freq = pd.infer_freq(date_series)
                summary["Date Info"] = {
                    "Date Column": date_col,
                    "Min Date": str(date_series.min()),
                    "Max Date": str(date_series.max()),
                    "Inferred Frequency": inferred_freq or "Irregular",
                }
        except Exception as e:
            summary["Date Info"] = {"Error": str(e)}

    product_summary = {}
    for product, group in df.groupby(product_col):
        group_dates = pd.to_datetime(group[date_col], errors="coerce").dropna()
        price_mean = group[price_col].mean() if price_col in group.columns else None
        product_summary[str(product)] = {
            "Start Date": str(group_dates.min()) if not group_dates.empty else None,
            "End Date": str(group_dates.max()) if not group_dates.empty else None,
            "Data Points": len(group),
            "Avg Price": float(price_mean) if price_mean is not None else None,
        }
    summary["Product Summary"] = product_summary

    return summary


def print_summary_table(summary):
    print(system_message("\n--- Basic Info ---"))
    basic_info = [
        ("Shape", summary.get("Shape")),
        ("Products", summary.get("Unique Products")),
    ]
    print(tabulate(basic_info, headers=["Metric", "Value"], tablefmt="grid"))

    print(system_message("\n--- Date Info ---"))
    date_info = summary.get("Date Info", {})
    if date_info:
        for k, v in date_info.items():
            print(system_message(f"{k}: {v}"))

    print(system_message("\n--- Missing Values ---"))
    missing = summary.get("Missing Values", {})
    if missing:
        rows = [(k, v["Count"], f"{v['Percent']:.2f}%") for k, v in missing.items()]
        print(
            tabulate(
                rows, headers=["Column", "Missing Count", "Missing %"], tablefmt="grid"
            )
        )

    print(system_message("\n--- Unique and Most Frequent ---"))
    unique_freq = summary.get("Unique and Most Frequent", {})
    rows = [(k, v["Unique Values"], v["Most Frequent"]) for k, v in unique_freq.items()]
    print(
        tabulate(
            rows, headers=["Column", "Unique Values", "Most Frequent"], tablefmt="grid"
        )
    )

    print(system_message("\n--- Product Summary ---"))
    prod_summary = summary.get("Product Summary", {})
    rows = [
        (k, v["Start Date"], v["End Date"], v["Data Points"], v["Avg Price"])
        for k, v in prod_summary.items()
    ]
    print(
        tabulate(
            rows,
            headers=["product ", "Start Date", "End Date", "Data Points", "Avg Price"],
            tablefmt="grid",
        )
    )


def run_data_summary(processed_config):
    print(system_message("📊 Data Summary Tool"))
    print(system_message(f"\n\nColumns detected: {list(processed_config['master_data'].columns)}"))


    date_col = input(system_input("\nEnter the name of the date column: ")).strip()
    product_col = input(system_input("\nEnter the name of the product column: ")).strip()
    price_col = input(system_input("\nEnter the name of the price column: ")).strip()

    summary = full_dataframe_summary(processed_config['master_data'], date_col, product_col, price_col)

    with open("memory/data_desc.txt", "w", encoding="utf-8") as f:
        f.write("data description:\n" + str(summary).strip())
    print(system_message("\n✅ Summary saved to memory/data_desc.txt\n"))
  
    # Display summary nicely in terminal
    print_summary_table(summary)
