"""
Insight Analyst
Role: Delivers a quick readout: What the data says before modeling begins
Responsibilities:
    - Generate descriptive summaries of the dataset (shape, products, dates, missing values).
    - Collect and validate column configuration from the user.
    - Categorize columns using LLM-based classification and human-in-the-loop approval.
    - Identify distinct products and key dimensions for modeling readiness.
"""


from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from agent_patterns.states import DataInsightState
from utils import utility
from utils.memory_handler import DataStore
from utils import theme_utility
from utils import chat_utility
import pandas as pd
from utils.theme_utility import console, log
import json
import os

DataValidationPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "DataValidationPrompt",
)


class DataAnalystAgent:
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        model: str,
        log_path: str = "logs/data_analyst.log",
    ):
        self.agent_name = agent_name
        self.agent_description = agent_description

        self.model = model
        self.llm = ChatOllama(model=self.model)
        self.log_path = log_path
        theme_utility.setup_console_logging(log_path)
        self.graph = self._build_graph()
    
    def Data_overview(self, state: DataInsightState):    
        log("[medium_purple3]LOG: Starting Data Summarizer...[/]")
        with console.status(f"[plum1] Data Summarizer Node setting up...[/]", spinner="dots"):
            sysprompt = 

# Pending : Because context should come from memory and REPL tools if required

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
        


    def ColumnCatogerizerNode(self, state: DataAnalystState):
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
                    role="user", message=f"LIST OF COLUMNS: {list(DataStore.get_df("master_data").columns)}"
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
            response = llm_structured.invoke(messages)
            theme_utility.display_response(response.thought_process, title="Thought")
            theme_utility.display_response(
                ", ".join(list(DataStore.get_df("master_data").columns)), title="ALL COLUMNS"
            )
        log("[dark_green]LOG: Column Catogories Generated[/]")
        asst_message = chat_utility.build_message_structure(role = "assistant", message = "Column Catogories Generated")
        return {
            "column_categories": dict(response.column_categories),
            "messages": [asst_message]
            }

    def colCatApprovalNode(self, state: DataAnalystState):
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

            user_feedback_llm = feedback_llm.invoke(message)
            log(f"[medium_purple3] Suggestion received[/] : [turquoise4]{suggestion}[/]")

        return {"user_feedback": user_feedback_llm}

    def _colCatDecisionNode(self, state: DataAnalystState):
        if state["user_feedback"].category == "approve":
            utility.save_in_memory(
                "column_categories", to_save=state["column_categories"]
            )
            return "approved"
        elif state["user_feedback"].category == "retry":
            return "retry"
        else:
            return "suggested_or_denied"

    def distinctProductNode(self, state: DataAnalystState):
        log("[medium_purple3]LOG: Identifying Distinct Products[/]")
        with console.status(
            f"[plum1] Distinct Product Identifier Node setting up...[/]", spinner="dots"
        ):
            unique_products = (
                DataStore.get_df("master_data")[state["column_categories"]["product_col"]]
                .dropna()
                .unique()
                .tolist()
            )
        console.print(f"\n[light_sky_blue1]Distinct Products:[/]")
        theme_utility.print_items_as_panels(unique_products)
        log("[dark_green]LOG: Distinct Product Identification completed[/]")
        asst_message = chat_utility.build_message_structure(role = "assistant", message = "Distinct Product Identification completed")
        return {
            "messages" : [asst_message],
            "distinct_products": unique_products,
            "completed":True
            }


    def _build_graph(self):
        g = StateGraph(DataAnalystState)
        g.add_node("DataSummaryNode", self.DataSummaryNode)
        g.add_node("colCategorizeNode", self.ColumnCatogerizerNode)
        g.add_node("colCatApprovalNode", self.colCatApprovalNode)
        g.add_node("distinctProductNode", self.distinctProductNode)

        g.add_edge(START, "DataSummaryNode")
        # g.add_edge("DataSummaryNode", "colCategorizeNode")
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
        g.add_edge("distinctProductNode", END)
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