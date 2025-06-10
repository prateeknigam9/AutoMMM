import os
from autommm.config.process_configuration import process_config
from autommm.config import configuration

from langchain_core.tools import tool
import pandas as pd
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated, List
import operator
from pathlib import Path

from IPython.display import Markdown, Image
from langchain_experimental.utilities import PythonREPL

from langgraph.graph import START, END, StateGraph
import subprocess


config = process_config(configuration)

df = config['master_data'] 
llm = config['llm']
data_description = config['data_description']
python310_executable = config['python310_executable']
data_profile_path = config['data_profile_path']

repl = PythonREPL()

@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Execute Python code using a Python REPL (Read-Eval-Print Loop).
    
    Args:
        code (str): The Python code to execute.
    
    Returns:
        str: The result of the executed code or an error message if execution fails.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


class ReportState(TypedDict):
    input : str
    overview : str
    sku_overview: Annotated[
        List, operator.add
        ]
    final_report : str
    formatted_report: str
    profile_report : str

def gen_data_profile(state: ReportState):
    """Run profile_report.py to generate a data profiling report using Python 3.10."""
    try:
        subprocess.check_call(["python", os.path.join("autommm","src","eda_agent","data_profiling.py")])
        # subprocess.call(['C:/Users/nigam/anaconda3/envs/agenticAI/python.exe', os.path.join("autommm","src","eda_agent","data_profiling.py")])
        profile_output = f"Data profiling report generated at: {data_profile_path}"
    except subprocess.CalledProcessError as e:
        profile_output = f"Failed to generate profiling report. Error: {repr(e)}"
    
    return {'profile_report': profile_output}

def data_overview(state: ReportState):
    backstory = """
    You are a knowledgeable market mix modelling expert, you work in a team as an analyst, you are trained to analyse the data.
    You directly report to business, so all the technical and statistical observations that you come up with, you translate them into business langugage,
    but do not avoid technical detail when appropriate.
    """
    instruction = """
    Generate a report in markdown format with the below format
    - Highlight the unique products in the data
    - The shape of the data
    - Split the columns as base, incremental (online and offline), external features, competitions, other features that can affect sale
    - Explain all columns as to business understanding"""

    # user_prompt = f"context data - {state['df']}"
    user_prompt = f"context data - {df}"

    overview = llm.invoke([
        {"role": "system", "content": backstory.strip()},
        {"role": "system", "content": instruction.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ])
    
    return {'overview' : overview.content}

def sku_overview(state: ReportState, sku:str):
    backstory = """
        You are a knowledgeable market mix modelling expert, you work in a team as an analyst, you are trained to analyse the data.
    You directly report to business, so all the technical and statistical observations that you come up with, you translate them into business langugage,
    but do not avoid technical detail when appropriate.
    """
    instruction = """
    Generate a report in markdown format with the below format
    - Sales Pattern throughout the timeline
    - Date distribution and spread
    - find outliers if any
        - mark  outliers
        - possible reason if any
        - impact on sales
        - suggestion to treat outliers
    - missing data if any    
        - percentage of missing points
        - if the missing points are in patches or irregular
        - mark  missing points
        - possible reason if any
        - impact on sales
        - suggestion to treat outliers
    - count of zeros in the data
        - if it is in patches
        - percentage of missings
    - Distinct values if column is categorical
    - Impact of each column on sales
    - Inter kpi correlations for all columns across all columns
    - Generate a Heatmap for correlation among kpis

    note : Round the numbers if required
    """
    # df = state['df']
    user_prompt = f"context data - {df[df['sku'] == sku]} for product - {sku}"

    sku_overview = llm.invoke([
        {"role": "system", "content": backstory.strip()},
        {"role": "system", "content": instruction.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ])
    return {'sku_overview' : [sku_overview.content]}

def data_vizualizer(state: ReportState, sku:str):
    llm_with_code_runner = llm.bind_tools([python_repl_tool])
    backstory = """
        You are a knowledgeable market mix modelling expert, you work in a team as an data vizualizer, you are trained to vizualize the data.
                """
    instruction = """
    Create clear and visually appealing charts using seaborn and plotly. Follow these rules:
    1. Add a title, labeled axes (with units), and a legend if needed.
    2. Use `sns.set_context("notebook")` for readable text and themes like `sns.set_theme()` or `sns.set_style("whitegrid")`.
    3. Use accessible color palettes like `sns.color_palette("husl")`.
    4. Choose appropriate plots: `sns.lineplot()`, `sns.barplot()`, or `sns.heatmap()`.
    5. Annotate key points (e.g., "Peak in 2020") for clarity.
    6. Ensure the chart's width and display resolution is no wider than 1000px.
    7. Display with `plt.show()`.
    """
    # df = state['df']
    user_prompt = f"""
    Generate charts for :
    - Sales Pattern throughout the timeline
    - find outliers if any
        - mark  outliers
        - possible reason if any
        - impact on sales
        - suggestion to treat outliers
    - missing data if any    
        - percentage of missing points
        - if the missing points are in patches or irregular
        - mark  missing points
        - possible reason if any
        - impact on sales
        - suggestion to treat outliers
    - count of zeros in the data
        - if it is in patches
        - percentage of missings
    - Distinct values if column is categorical
    - Impact of each column on sales
    - Inter kpi correlations for all columns across all columns
    - Generate a Heatmap for correlation among kpis
    
    context data - {df[df['sku'] == sku]} for product - {sku}
    """

    sku_overview = llm.invoke([
        {"role": "system", "content": backstory.strip()},
        {"role": "system", "content": instruction.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ])
    return {'sku_overview' : [sku_overview.content]}

def aggregator(state: ReportState):

    combined_sku_reports = "\n\n --- \n\n".join(state['sku_overview'])
    combined_report = f"""
    Data Overview : {state['overview']}
    product based analysis : {combined_sku_reports}
    """
    return {'final_report' : combined_report}

def formatter(state: ReportState):
    backstory = """
    You are an expert technical writer and Markdown formatter. A detailed data analysis report has been generated, 
    but it is currently in a raw text format with escaped newlines (`\n`), inconsistent structure, and minimal formatting.
    This report is meant to be shared with stakeholders and collaborators. Therefore, it needs to be reformatted for clarity, professionalism, and readability — using Markdown.
    """
    instruction = """
    Your task is to convert the raw text report into a properly structured Markdown document.

    Apply the following formatting guidelines:
    1. Convert `\\n` into actual line breaks.
    2. Use `#`, `##`, `###` for main sections, subsections, and sub-subsections.
    3. Preserve or format bold text (`**bold**`) where emphasis is needed.
    4. Format any references to data columns, KPIs, or variable names using backticks (`` ` ``).
    5. Convert lists into bullet points (`-`, `*`) with proper indentation.
    6. Where appropriate, convert structured content into **Markdown tables** (especially lists of columns, grouped features, or comparisons).
    7. Where applicable, use simple text-based diagrams or layouts for clarity.
    8. Ensure the final output is clean, readable, and professional in a Markdown viewer (e.g., GitHub, Notion, Jupyter).

    Be consistent and logical in how content is grouped.

    """
    context = f"""
    Below is the raw report content that needs to be formatted:
    {state['final_report']}
    """


    markdown_formatted_report = llm.invoke([
        {"role": "system", "content": backstory.strip()},
        {"role": "system", "content": instruction.strip()},
        {"role": "user", "content": context.strip()}
    ])

    return {'formatted_report' : markdown_formatted_report}


eda_report_builder = StateGraph(ReportState)

eda_report_builder.add_node("data_overview", data_overview)
eda_report_builder.add_node("gen_data_profile", gen_data_profile)
eda_report_builder.add_node("sku_overview_a", lambda state: sku_overview(state, sku='sku_a'))
eda_report_builder.add_node("sku_overview_b", lambda state: sku_overview(state, sku='sku_b'))
eda_report_builder.add_node("sku_overview_c", lambda state: sku_overview(state, sku='sku_c'))
eda_report_builder.add_node("aggregator",aggregator)
eda_report_builder.add_node("formatter",formatter)

eda_report_builder.add_edge(START,"data_overview")
eda_report_builder.add_edge(START,"gen_data_profile")
eda_report_builder.add_edge(START,"sku_overview_a")
eda_report_builder.add_edge(START,"sku_overview_b")
eda_report_builder.add_edge(START,"sku_overview_c")

eda_report_builder.add_edge("data_overview","aggregator")
eda_report_builder.add_edge("sku_overview_a","aggregator")
eda_report_builder.add_edge("sku_overview_b","aggregator")
eda_report_builder.add_edge("sku_overview_c","aggregator")
eda_report_builder.add_edge("aggregator","formatter")

eda_report_builder.add_edge("formatter", END)
eda_report_builder.add_edge("gen_data_profile", END)

graph = eda_report_builder.compile()

# display(Image(eda_report.get_graph().draw_mermaid_png()))


