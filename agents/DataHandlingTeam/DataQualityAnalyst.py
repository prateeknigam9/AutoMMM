"""
Data Quality Analyst
Role: Validates the input dataset at brand and product levels and generates a final data quality report for modeling readiness.
Responsibilities:
    - Run automated checks on structure, formats, types, and completeness at the brand level.
    - Perform detailed validation across all distinct products using custom tools.
    - Summarize tool outputs into LLM-generated insights for each product.
    - Compile a markdown-based validation report with structured summaries and actionable insights.
"""
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from agent_patterns.states import DataQualityAnalystState
from agent_patterns.tool_calling_agent.tool_agent import ToolAgent
from tools.tools_data_analysis import (
    generate_validation_summary,
    data_describe,
    validate_column_name_format,
    validate_date_format,
    validate_data_types,
    duplicate_checker,
    validate_time_granularity,
    raise_validation_warnings,
)
from tools.tools_data_analysis import (
    validate_missing_dates,
    validate_nulls,
    validate_duplicate_rows,
    validate_unique_time_product,
    validate_price_consistency,
    validate_outliers,
    validate_minimum_duration,
    validate_media_spend_coverage,
    validate_oos_sales,
    validate_seasonality,
    validate_media_sales_sync,
)

from utils import utility
import ast
from utils import theme_utility
from utils import chat_utility
from utils.memory_handler import DataStore
from utils.theme_utility import console, log
import subprocess
import pandas as pd
import os

DataValidationPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "DataValidationPrompt",
)

data_quality_analyst_messages = utility.load_prompt_config(
    r"prompts\user_messages.yaml",
    "data_quality_analyst",
)

class DataQualityAnalystAgent:
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        model: str,
        log_path: str = "logs/agent.log",
    ):
        self.agent_name = agent_name
        self.agent_description = agent_description

        self.model = model
        self.llm = ChatOllama(model=self.model)
        self.log_path = log_path
        theme_utility.setup_console_logging(log_path)
        self.graph = self._build_graph()

    def dataProfilerNode(self, state: DataQualityAnalystState):
        python310_executable = chat_utility.take_user_input("python310_executable path")
        df = DataStore.get_df("master_data")
        csv_path = "output/temp_master_data.csv"
        os.makedirs("output", exist_ok=True)
        df.to_csv(csv_path, index=False)
        try:
            import ydata_profiling
        except ImportError:
            subprocess.check_call([python310_executable, "-m", "pip", "install", "ydata-profiling","--quiet"])
            subprocess.check_call([python310_executable, "-m", "pip", "install", "openpyxl","--quiet"])

        code = data_quality_analyst_messages['data_profiling_code'].format(
            csv_path = csv_path,
            data_profile_path = "output/data_profile_report.html"
        )
        try:
            # Run the code using subprocess
            subprocess.check_call([python310_executable, "-c", code])
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with exit code {e.returncode}: {e}")
        except FileNotFoundError:
            print(f"Error: Python executable not found at {python310_executable}")
        except Exception as e:
            print(f"Error: {str(e)}")
        if os.path.exists(csv_path):
            os.remove(csv_path)


    def toolRunnerDataLevelNode(self, state: DataQualityAnalystState):
        log("[medium_purple3]LOG: Running tools to generate data validation report at[/] [turquoise4]Brand Level[/]")
        with console.status(
            f"[plum1] Data Loading Node setting up...[/]\n", spinner="dots"
        ):
            print("")
        agent = ToolAgent(
            agent_name="Data Validation Agent",
            agent_description="Agent to validate MMM dataset before modeling",
            agent_goal="validate the data",
            tools=[
                generate_validation_summary,
                data_describe,
                validate_column_name_format,
                validate_date_format,
                validate_data_types,
                duplicate_checker,
                validate_time_granularity,
                raise_validation_warnings,
            ],
            model=self.model,
            log_path=self.log_path,
        )
        prompt = DataValidationPrompt["toolRunnerDataLevel"].strip()
        result = agent.graph_invoke(prompt)

        validation_report = {
            "product_id": "brand",
            "tool_outputs": result['tool_response_list'],
            "llm_summary": result['finalresponse']
        }
        log("[dark_green]LOG: Data validation report Generated at[/] [turquoise4]Brand Level[/]")
        asst_message = chat_utility.build_message_structure(role = "assistant", message = "Data validation report Generated")
            
        return {
            "tool_results":[validation_report],
            "messages": [asst_message]
            }

    def toolRunnerProductLevelNode(self, state: DataQualityAnalystState):
        log("[medium_purple3]LOG: Running tools to generate data validation report at[/] [turquoise4]Product Level[/]")
        validation_tools = [
            validate_missing_dates,
            validate_nulls,
            validate_duplicate_rows,
            validate_unique_time_product,
            validate_price_consistency,
            validate_outliers,
            validate_minimum_duration,
            validate_media_spend_coverage,
            validate_oos_sales,
            validate_seasonality,
            validate_media_sales_sync,
        ]
        all_results = []
        responses = {}
        distinct_products = DataStore.get_str('distinct_products')
        distinct_products = ast.literal_eval(distinct_products)
        for product_id in distinct_products:
            tool_outputs = {}
            for tool in validation_tools:
                try:
                    result = tool.invoke({"product_id": product_id})
                    tool_outputs[tool.name] = result
                except Exception as e:
                    tool_outputs[tool.name] = f"Error: {str(e)}"

            prompt = self.format_tool_outputs_for_prompt(product_id, tool_outputs)
            message = chat_utility.build_message_structure(role="user", message=prompt)
            response = self.llm.invoke([message])
            summary = response.content
            responses[product_id] = summary
            theme_utility.display_response(summary, title = f"LLM Summary for {product_id}")
            all_results.append({
                "product_id": product_id,
                "tool_outputs": tool_outputs,
                "llm_summary": summary
            })

        state["tool_results"].extend(all_results)
        log("[dark_green]LOG: Data validation report Generated at[/] [turquoise4]Product Level[/]")
        return state

    def finalReportGeneratorNode(self, state: DataQualityAnalystState):
        log("[medium_purple3]LOG: Starting final data validaiton report Generation[/]")
        with console.status("[plum1] Generating final markdown report using LLM...", spinner="dots"):
            all_summaries = []

            for entry in state["tool_results"]:
                product_id = entry.get("product_id", "unknown")
                summary = entry.get("llm_summary", "")
                all_summaries.append(f"## Product: `{product_id}`\n\n{summary.strip()}\n")

            markdown_input = "\n".join(all_summaries)

            prompt = f"""
            You are a data QA analyst creating a final data validation report for market mix modeling.

            Below are summaries for each product. Format these into a clean, readable **Markdown** report. Use:
            - A title
            - Subsections per product
            - Bold headers for tool outputs or issues
            - Bullet points or tables where appropriate

            Summaries:
            {markdown_input}
            """

            message = chat_utility.build_message_structure(
                role="user",
                message=prompt,
            )

            response = self.llm.invoke([message])
            markdown_report = response.content

            report_path = 'final_llm_report.txt'
            utility.save_to_memory_file(report_path, str(markdown_report))

            theme_utility.display_response(markdown_report[:2000], title="LLM Markdown Report (Preview)")
            log(f"[dark_green] Markdown report saved to[/] [turquoise4]{report_path}[/]")
            asst_message = chat_utility.build_message_structure(role = "assistant", message = f"Markdown report saved to {report_path}")
            return {
                "qa_report": markdown_report,
                "qa_report_path": report_path,
                "messages": [asst_message],
                "completed":True
                 }



    def _build_graph(self):
        g = StateGraph(DataQualityAnalystState)
        g.add_node("dataProfilerNode", self.dataProfilerNode)
        g.add_node("toolRunnerDataLevelNode", self.toolRunnerDataLevelNode)
        g.add_node("toolRunnerProductLevelNode", self.toolRunnerProductLevelNode)
        g.add_node("finalReportGeneratorNode", self.finalReportGeneratorNode)

        g.add_edge(START, "dataProfilerNode")
        g.add_edge("dataProfilerNode", "toolRunnerDataLevelNode")
        g.add_edge("toolRunnerDataLevelNode", "toolRunnerProductLevelNode")
        g.add_edge("toolRunnerProductLevelNode", "finalReportGeneratorNode")
        g.add_edge("finalReportGeneratorNode", END)
        return g.compile(name=self.agent_name)


    def format_tool_outputs_for_prompt(self, product_id, tool_results):
        prompt = DataValidationPrompt['toolRunnerProductLevel'].format(product_id = product_id)
        for tool_name, result in tool_results.items():
            prompt += f"\nTool: {tool_name}\nResult: {result}\n"
        prompt += "\nSummarize key data quality issues, potential impact on modeling, and what actions may be needed."
        return prompt
