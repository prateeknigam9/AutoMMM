from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from datetime import datetime
from itertools import zip_longest
from agent_patterns.states import DataValidationState, Feedback
from agent_patterns.structured_response import (
    ColumnCategories,
    ColumnCategoriesResponse,
    TypeValidationResponse,
)
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
from utils.memory_handler import DataStore
from utils import theme_utility
from utils import chat_utility
import pandas as pd
from pathlib import Path
from typing import List
from utils.theme_utility import console, log
from rich import print, prompt
from tabulate import tabulate
import json
import os

DataValidationPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "DataValidationPrompt",
)


class DataValidationAgent:
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        tools: list,
        model: str,
        df: pd.DataFrame,
        log_path: str = "logs/agent.log",
    ):
        self.agent_name = agent_name
        self.agent_description = agent_description

        self.tools = tools
        self.tool_desc, self.tool_names = utility.tools_to_action_prompt(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.model = model
        self.llm = ChatOllama(model=self.model)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.df = df
        self.log_path = log_path
        theme_utility.setup_console_logging(log_path)
        self.graph = self._build_graph()

    async def graph_invoke(self, query: str):
        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=query)
            ]
        }
        theme_utility.print_startup_info(
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            is_interactive=True,
        )
        result = await self.graph.ainvoke(inputs)
        return result

    async def loadingDataNode(self, state: DataValidationState):
        log("[medium_purple3]LOG: Starting Data loading[/]")
        with console.status(
            f"[plum1] Data Loading Node setting up...[/]\n", spinner="dots"
        ):
            console.print("")
        file_path = chat_utility.take_user_input("Enter the path to your Excel file:")
        while not os.path.isfile(file_path):
            file_path = chat_utility.take_user_input(
                "Invalid path. Please re-enter Excel file path: "
            ).strip()
        excel_file = pd.ExcelFile(file_path)
        console.print("[sandy_brown]sheet_names:[/] ", excel_file.sheet_names)
        sheet_name = chat_utility.take_user_input("[sandy_brown]Sheet:[/] ").strip()
        while sheet_name not in excel_file.sheet_names:
            sheet_name = chat_utility.take_user_input(
                "Invalid sheet name. Please re-enter: "
            ).strip()

        DataStore.set_df("master_data", pd.read_excel(file_path, sheet_name))
        log(f"[medium_purple3] saving master data in memory with key[/] - [turquoise4]master_data[/]")

        data_context = chat_utility.take_user_input(
            "Please describe the context of the data (e.g., what is the data about, business context, etc.): "
        )

        DataStore.set_str("data_context", data_context)
        log(f"[medium_purple3] saving data context in memory with key[/] - [turquoise4]data_context[/]")

        response = chat_utility.build_message_structure(
            role="assistant", message="data and context data loaded"
        )
        log("[dark_green]LOG: Data Loading completed[/]")
        return {"message": [response]}

    async def ColumnContextExtractNode(self, state: DataValidationState):
        log("[medium_purple3]LOG: Starting Column Context Extraction[/]")
        with console.status(
            f"[plum1] Column Context Extraction Node setting up...[/]\n", spinner="dots"
        ):
            master_data = DataStore.get_df("master_data")
            data_context = DataStore.get_str("data_context")
        prompt = DataValidationPrompt["ColumnContextExtraction"]
        prompt += f"\nContext: {data_context}"
        prompt += "\n\nPlease respond only with a valid JSON dictionary."
        message = chat_utility.build_message_structure(role="system", message=prompt)
        messages = [message] + [
            chat_utility.build_message_structure(
                role="user",
                message=f"\nData sample (first 5 rows): {master_data.head(5).to_dict(orient='records')}",
            )
        ]
        with console.status(f"[plum1] Generating response for column context...[/]\n", spinner="dots"):
            response = await self.llm.ainvoke(messages)
        log(f"[medium_purple3]LOG: Generated column context response[/]")
        try:
            theme_utility.print_dictionary(
                json.loads(response.content), title="Column Context"
            )
        except:
            theme_utility.display_response(response.content)

        os.makedirs("memory", exist_ok=True)
        approved, _ = chat_utility.ask_user_approval(agent_name="Column Context")
        if approved is True:
            utility.save_to_memory_file('data_context.txt', str(data_context))
            utility.save_to_memory_file('column_context.txt', str(response.content).strip())
            log(f"[medium_purple3]LOG: Saved column context to memory[/]")
        else:
            console.print(
                "\n[sandy_brown]Opening the text file... Please edit and save, then return here.[/bold yellow]"
            )
            os.startfile(r"memory\column_context.txt")
            chat_utility.take_user_input(
                "\n[bold blue]Press ENTER when you're done editing the text file[/bold blue]"
            ).strip()
        
        with open(r"memory\column_context.txt", "r", encoding="utf-8") as f:
            updated_text = f.read()
        DataStore.set_str("data_context", updated_text)
        log(f"[medium_purple3]LOG: Saved data context to memory[/]")
        log("[dark_green]LOG: Column Context Extraction completed[/]")
        return {"message": [updated_text]}

    async def DataSummaryNode(self, state: DataValidationState):
        log("[medium_purple3]LOG: Starting Data Summarizer...[/]")
        with console.status(
            f"[plum1] Data Summarizer Node setting up...[/]", spinner="dots"
        ):
            col_config = [
                "date_col",
                "product_col",
                "price_col",
                "revenue_col",
                "units_sold_col",
                "oos_col",
            ]
            actual_cols = list(DataStore.get_df("master_data").columns)
            rows = list(zip_longest(col_config, actual_cols))
            required_columns_df = pd.DataFrame(
                rows, columns=["COLUMN_CONFIG", "COLUMN_NAME"]
            )
            file_path = r"user_inputs\column_config.xlsx"
        chat_utility.user_input_excel(required_columns_df, file_path=file_path)

        console.print(f"\n[light_sky_blue1]Columns detected: [/]")
        theme_utility.print_items_as_panels(actual_cols)

        column_config_df = pd.read_excel(file_path)
        DataStore.set_df("column_config_df", column_config_df)
        column_config = dict(
            zip(column_config_df["COLUMN_CONFIG"], column_config_df["COLUMN_NAME"])
        )

        date_col = column_config["date_col"]
        product_col = column_config["product_col"]
        price_col = column_config["price_col"]

        data_summary = self.full_dataframe_summary(
            DataStore.get_df("master_data"), date_col, product_col, price_col
        )

        # --- BASIC INFO ---
        console.print("\n[bold underline]Basic Info[/bold underline]")
        basic_info = [
            ("Shape", data_summary.get("Shape")),
            ("Products", data_summary.get("Unique Products")),
        ]
        theme_utility.print_rich_table(basic_info, headers=["Metric", "Value"])

        # --- DATE INFO ---
        date_info = data_summary.get("Date Info", {})
        if date_info:
            console.print("\n[bold underline]Date Info[/bold underline]")
            theme_utility.print_rich_table(
                list(date_info.items()), headers=["Metric", "Value"]
            )

        # --- MISSING VALUES ---
        missing = data_summary.get("Missing Values", {})
        if missing:
            console.print("\n[bold underline]Missing Values[/bold underline]")
            rows = [(k, v["Count"], f"{v['Percent']:.2f}%") for k, v in missing.items()]
            theme_utility.print_rich_table(
                rows, headers=["Column", "Missing Count", "Missing %"]
            )

        # --- UNIQUE & MOST FREQUENT ---
        unique_freq = data_summary.get("Unique and Most Frequent", {})
        if unique_freq:
            console.print("\n[bold underline]Unique and Most Frequent[/bold underline]")
            rows = [
                (k, v["Unique Values"], v["Most Frequent"])
                for k, v in unique_freq.items()
            ]
            theme_utility.print_rich_table(
                rows, headers=["Column", "Unique Values", "Most Frequent"]
            )

        # --- PRODUCT SUMMARY ---
        prod_summary = data_summary.get("Product Summary", {})
        if prod_summary:
            console.print("\n[bold underline]Product Summary[/bold underline]")
            rows = [
                (k, v["Start Date"], v["End Date"], v["Data Points"], v["Avg Price"])
                for k, v in prod_summary.items()
            ]
            theme_utility.print_rich_table(
                rows,
                headers=[
                    "Product",
                    "Start Date",
                    "End Date",
                    "Data Points",
                    "Avg Price",
                ],
            )
        utility.save_to_memory_file('data_summary.txt', str(data_summary))
        log(f"[medium_purple3]LOG: Saved data summary to memory[/]")
        log("[dark_green]LOG: Data generated report [/]")
        return {"data_summary": data_summary}

    async def ColumnCatogerizerNode(self, state: DataValidationState):
        log("[medium_purple3]LOG: Starting Column Catogorizing[/]")
        with console.status(
            f"[plum1] Column Catogorizer Node setting up...[/]", spinner="dots"
        ):
            llm_structured = self.llm.with_structured_output(ColumnCategoriesResponse)
            messages = chat_utility.build_message_structure(
                role="system", message=DataValidationPrompt["ColumnCatogerizer"]
            )
            messages = [messages] + [
                chat_utility.build_message_structure(
                    role="user", message=f"LIST OF COLUMNS: {list(self.df.columns)}"
                )
            ]
            try:
                if state["user_feedback"].category == "retry":
                    chat_utility.append_to_structure(
                        history=messages,
                        role="assistant",
                        message=f"THOUGHT : {state['user_feedback'].thought}",
                    )
                    chat_utility.append_to_structure(
                        history=messages,
                        role="user",
                        message=f"FEEDBACK: {state['user_feedback'].feedback}",
                    )
            except:
                pass
            response = await llm_structured.ainvoke(messages)
            theme_utility.display_response(response.thought_process, title="Thought")
            theme_utility.display_response(
                ", ".join(list(self.df.columns)), title="ALL COLUMNS"
            )
        log("[dark_green]LOG: Column Catogories Generated[/]")
        return {"column_categories": dict(response.column_categories)}

    async def colCatApprovalNode(self, state: DataValidationState):
        log("[medium_purple3]LOG: Column Category Approval required[/]")
        with console.status(
            f"[plum1] Column Category Approval Node setting up...[/]", spinner="dots"
        ):
            console.print(f"\n[light_sky_blue1]Printing column categories:[/]\n")
            theme_utility.print_dictionary(
                state["column_categories"], title="column_categories"
            )
        approved, suggestion = chat_utility.ask_user_approval(
            agent_name="Category Node"
        )
        if approved is True:
            user_feedback_llm = Feedback(category="approve", feedback="", thought="")
            log(f"[medium_purple3] Category Approval Node approved, printing data dictionary[/]")
        elif approved is False:
            user_feedback_llm = Feedback(category="retry", feedback="", thought="")
            log(f"[red3] Category Approval Node Denied, retrying...[/]")
        else:
            feedback_llm = self.llm.with_structured_output(Feedback)
            message = chat_utility.build_message_structure(
                role="system", message=DataValidationPrompt["ApprovalNode"]
            )
            message = [message] + [
                chat_utility.build_message_structure(role="user", message=suggestion)
            ]

            user_feedback_llm = await feedback_llm.ainvoke(message)
            log(f"[medium_purple3] Suggestion received[/] : [turquoise4]{suggestion}[/]")

        return {"user_feedback": user_feedback_llm}

    def _colCatDecisionNode(self, state: DataValidationState):
        if state["user_feedback"].category == "approve":
            utility.save_in_memory(
                "column_categories", to_save=state["column_categories"]
            )
            return "approved"
        elif state["user_feedback"].category == "retry":
            return "retry"
        else:
            return "suggested_or_denied"

    async def distinctProductNode(self, state: DataValidationState):
        log("[medium_purple3]LOG: Identifying Distinct Products[/]")
        with console.status(
            f"[plum1] Distinct Product Identifier Node setting up...[/]", spinner="dots"
        ):
            unique_products = (
                self.df[state["column_categories"]["product_col"]]
                .dropna()
                .unique()
                .tolist()
            )
        console.print(f"\n[light_sky_blue1]Distinct Products:[/]")
        theme_utility.print_items_as_panels(unique_products)
        log("[dark_green]LOG: Distinct Product Identification completed[/]")
        return {"distinct_products": unique_products}

    async def toolRunnerDataLevelNode(self, state: DataValidationState):
        log("[medium_purple3]LOG: Running tools to generate data validation report at[/] [turquoise4]Brand Level[/]")
        with console.status(
            f"[plum1] Data Loading Node setting up...[/]\n", spinner="dots"
        ):
            print("")
        agent = ToolAgent(
            agent_name="Data Validation Agent",
            agent_description="Agent to validate MMM dataset before modeling",
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
        result = await agent.graph_invoke_async(prompt)

        validation_report = {
            "product_id": "brand",
            "tool_outputs": result['tool_response_list'],
            "llm_summary": result['finalresponse']
        }
        log("[dark_green]LOG: Data validation report Generated at[/] [turquoise4]Brand Level[/]")
        return {"tool_results":[validation_report]}

    def format_tool_outputs_for_prompt(self, product_id, tool_results):
        prompt = DataValidationPrompt['toolRunnerProductLevel'].format(product_id = product_id)
        for tool_name, result in tool_results.items():
            prompt += f"\nTool: {tool_name}\nResult: {result}\n"
        prompt += "\nSummarize key data quality issues, potential impact on modeling, and what actions may be needed."
        return prompt

    async def toolRunnerProductLevelNode(self, state: DataValidationState):
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

        for product_id in state['distinct_products']:
            tool_outputs = {}
            for tool in validation_tools:
                try:
                    result = tool.invoke({"product_id": product_id})
                    tool_outputs[tool.name] = result
                except Exception as e:
                    tool_outputs[tool.name] = f"Error: {str(e)}"

            prompt = self.format_tool_outputs_for_prompt(product_id, tool_outputs)
            message = chat_utility.build_message_structure(role="user", message=prompt)
            response = await self.llm.ainvoke([message])
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

    async def finalReportGeneratorNode(self, state: DataValidationState):
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

            response = await self.llm.ainvoke([message])
            markdown_report = response.content

            report_path = 'final_llm_report.txt'
            utility.save_to_memory_file(report_path, str(markdown_report))

            theme_utility.display_response(markdown_report[:2000], title="LLM Markdown Report (Preview)")
            log(f"[dark_green] Markdown report saved to[/] [turquoise4]{report_path}[/]")

            return {"final_report": markdown_report, "report_path": report_path}


    def _build_graph(self):
        g = StateGraph(DataValidationState)
        g.add_node("loadingDataNode", self.loadingDataNode)
        g.add_node("ColumnContextExtractNode", self.ColumnContextExtractNode)
        g.add_node("DataSummaryNode", self.DataSummaryNode)
        g.add_node("colCategorizeNode", self.ColumnCatogerizerNode)
        g.add_node("colCatApprovalNode", self.colCatApprovalNode)
        g.add_node("distinctProductNode", self.distinctProductNode)
        g.add_node("toolRunnerDataLevelNode", self.toolRunnerDataLevelNode)
        g.add_node("toolRunnerProductLevelNode", self.toolRunnerProductLevelNode)
        g.add_node("finalReportGeneratorNode", self.finalReportGeneratorNode)

        g.add_edge(START, "loadingDataNode")
        g.add_edge("loadingDataNode", "ColumnContextExtractNode")
        g.add_edge("ColumnContextExtractNode", "DataSummaryNode")
        g.add_edge("DataSummaryNode", "colCategorizeNode")
        g.add_edge("colCategorizeNode", "colCatApprovalNode")
        g.add_conditional_edges(
            "colCatApprovalNode",
            self._colCatDecisionNode,
            {
                "approved": "distinctProductNode",
                "retry": "colCategorizeNode",
                "suggested_or_denied": "colCatApprovalNode",
            },
        )
        g.add_edge("distinctProductNode", "toolRunnerDataLevelNode")
        g.add_edge("toolRunnerDataLevelNode", "toolRunnerProductLevelNode")
        g.add_edge("toolRunnerProductLevelNode", "finalReportGeneratorNode")
        g.add_edge("finalReportGeneratorNode", END)
        return g.compile(name=self.agent_name)

    def full_dataframe_summary(self, df, date_col, product_col, price_col):
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
