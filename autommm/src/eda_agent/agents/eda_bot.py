import os
from autommm.config.process_configuration import process_config
from autommm.config import configuration

from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph import START, END, StateGraph
import subprocess
import operator
import yaml

config = process_config(configuration)

df = config["master_data"]
llm = config["llm"]
data_description = config["data_description"]
python310_executable = config["python310_executable"]
data_profile_path = config["data_profile_path"]

prompts_config_path = os.path.join("autommm", "src", "eda_agent", "prompts_config.yaml")
def load_prompt_config(path: str, key: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)[key]


class ReportState(TypedDict):
    input: str
    overview: str
    sku_overview: Annotated[List, operator.add]
    final_report: str
    formatted_report: str
    profile_report: str



def gen_data_profile(state: ReportState):
    """Run profile_report.py to generate a data profiling report using Python 3.10."""
    print("--- Node: Generating data profile report... ---")
    try:
        subprocess.check_call(
            ["python", os.path.join("autommm", "src", "eda_agent", "data_profiling.py")]
        )
        # subprocess.call(['C:/Users/nigam/anaconda3/envs/agenticAI/python.exe', os.path.join("autommm","src","eda_agent","data_profiling.py")])
        profile_output = f"Data profiling report generated at: {data_profile_path}"
    except subprocess.CalledProcessError as e:
        profile_output = f"Failed to generate profiling report. Error: {repr(e)}"

    print(f"Data profiling status: {profile_output}")
    return {"profile_report": profile_output}


def data_overview(state: ReportState):
    print("--- Node: Generating overall data overview... ---")
    prompt_config = load_prompt_config(
        prompts_config_path, "eda_analyst_agent"
    )

    user_prompt = prompt_config["template"].format(
        sample_df=df, data_description=data_description
    )
    messages = [
        {"role": "system", "content": f"goal: {prompt_config['goal']}"},
        {"role": "system", "content": f"backstory: {prompt_config['backstory']}"},
        {"role": "system", "content": f"instruction: {prompt_config['instruction']}"},
        {"role": "user", "content": user_prompt},
    ]
    try:
        overview = llm.invoke(messages)
        print("Overall data overview generated.")
        return {"overview": overview.content}
    except Exception as e:
        print(f"Error in DataOverview node: {e}")
        return {"overview": f"Failed to generate data overview: {e}"}


def sku_overview(state: ReportState, product: str):
    print(f"--- Node: Generating SKU overview for {product}... ---")
    prompt_config = load_prompt_config(
        prompts_config_path, "product_analyst_agent"
    )
    user_prompt = prompt_config["template"].format(
        product=product, context_data=df[df["sku"] == product]
    )
    messages = [
        {"role": "system", "content": f"goal: {prompt_config['goal']}"},
        {"role": "system", "content": f"backstory: {prompt_config['backstory']}"},
        {"role": "system", "content": f"instruction: {prompt_config['instruction']}"},
        {"role": "user", "content": user_prompt},
    ]
    try:
        product_response = llm.invoke(messages)
        print(f"SKU analysis complete for '{product}'.")
        return {"sku_overview": [product_response.content]}
    except Exception as e:
        print(f"Error in SKU analysis for '{product}': {e}")
        return {
            "sku_overview": [f"Failed to generate SKU analysis for '{product}': {e}"]
        }


def aggregator(state: ReportState):
    print("--- Node: Combining all report sections... ---")
    combined_sku_reports = "\n\n --- \n\n".join(state["sku_overview"])
    combined_report = f"""
    Data Overview : {state['overview']}
    product based analysis : {combined_sku_reports}
    """

    print("Raw report sections aggregated.")
    return {"final_report": combined_report}


def formatter(state: ReportState):
    print("--- Node: Formatting all report sections... ---")
    prompt_config = load_prompt_config(
        prompts_config_path, "markdown_formatter_agent"
    )
    user_prompt = prompt_config["template"].format(final_report=state["final_report"])
    messages = [
        {"role": "system", "content": f"goal: {prompt_config['goal']}"},
        {"role": "system", "content": f"backstory: {prompt_config['backstory']}"},
        {"role": "system", "content": f"instruction: {prompt_config['instruction']}"},
        {"role": "user", "content": user_prompt},
    ]
    try:
        markdown_formatted_report = llm.invoke(messages)
        print("Final report formatted to Markdown.")
        return {"formatted_report": markdown_formatted_report.content}
    except Exception as e:
        print(f"Error in Formatter node: {e}")
        return {"formatted_report": f"Failed to format report: {e}"}
