from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, Literal

from agents.utils.data_analysis_tools import DataOperations
from agents.utils import utiltiy

import os
from datetime import datetime
import pandas as pd

from utils import theme_and_color, console
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich.text import Text
from rich.style import Style
from datetime import datetime
import time

AgentPrompts = utiltiy.load_prompt_config(
    r"C:\Users\nigam\Documents\AutoMMM\agents\prompts\data_analysis_agent.yaml",
    "AgentPrompts",
)

class DataValidationState(TypedDict):
    input_path : str
    sheet_name : str 
    output_path : str 
    df : pd.DataFrame
    column_categories: dict
    distinct_products : list
    user_feedback : None
    node_to_update: str
    data_val_report : str
    
class DataValidationAgent:
    def __init__(self, agent_name: str, model_name:str, verbose:bool=False):
        self.agent_name = agent_name
        self.llm = ChatOllama(model = model_name, temperature=0)
        self.graph = self._build_graph()
        self.memory_path = r'C:\Users\nigam\Documents\AutoMMM\memory'
        self.verbose = verbose
        console.log(f"[light_sky_blue1]▸ {agent_name} initialization started[/]")

    def run(self):
        init_msg = {
        'input_path': r'C:\Users\nigam\Documents\AutoMMM\data_to_model.xlsx',
        'sheet_name': 'Sheet1',
        'output_path': 'output.csv'
        }
        theme_and_color.print_startup_info(
            agent_name = "DataValidator",
            agent_description= "Validates and cleans data before analysis.",
            is_interactive = True
        )
        return self.graph.invoke(init_msg)

    def _loaderNode(self, state: DataValidationState):
        with console.status(f"[plum1] Loader Node setting up...[/]", spinner="dots"):
            df = pd.read_excel(state['input_path'], sheet_name = state['sheet_name'])
            console.print(f"\n[light_sky_blue1]Printing data head:[/]\n")
            console.print(df.head())
        return {"df": df}
    
    def _colCategorizeNode(self, state: DataValidationState):
        with console.status(f"[plum1] Column Catogorizer Node setting up...[/]", spinner="dots"):
            history = []
            try: 
                if state['user_feedback'].category == 'retry':
                    history.append({'role':'assistant', 'content': state['user_feedback'].thought})
                    history.append({'role':'user', 'content': state['user_feedback'].feedback})
            except:
                pass
            self.operations = DataOperations(state['df'])
            column_categories = self.operations.categorize_columns(history)
        return {'column_categories': column_categories}

    def _colCatApprovalNode(self, state: DataValidationState):
        with console.status(f"[plum1] Column Category Approval Node setting up...[/]", spinner="dots"):
            console.print(f"\n[light_sky_blue1]Printing column categories:[/]\n")
            theme_and_color.print_dict_nicely(state['column_categories'],'column_categories')  
            
        class Feedback(BaseModel):
            category: Literal['approve', 'retry', 'retry with suggestion']
            feedback : str = Field(..., description="user response")
            thought: str = Field(..., description="Reasoning about what needs to be done next")

        approved, suggestion = theme_and_color.ask_user_approval("Category Approval Node")
        if approved is True:
            user_feedback_llm = Feedback(
                category = "approve",
                feedback = "",
                thought = ""
            )
            console.log(f"[plum1]✔ Category Approval Node approved, printing data dictionary[/]")
        elif approved is False:
            user_feedback_llm = Feedback(
                category = "retry",
                feedback = "",
                thought = ""
            )
            console.log(f"[red3]✘ Category Approval Node Denied, retrying...[/]")

        else:
            feedback_llm = self.llm.with_structured_output(Feedback)
            system_prompt = AgentPrompts["ApprovalNode"]
            messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': suggestion}]
            user_feedback_llm = feedback_llm.invoke(messages)
            console.log(f"[medium_purple3]✏ Suggestion received: {suggestion}[/]")

        return {'user_feedback': user_feedback_llm}

    def _colCatDecisionNode(self, state: DataValidationState):
        if state['user_feedback'].category == 'approve':
            return "approved"
        elif state['user_feedback'].category == 'retry':
            return "retry"
        else:
            return "suggested_or_denied"

    def _distinctProductNode(self, state: DataValidationState):
        with console.status(f"[plum1] Distinct Product Identifier Node setting up...[/]", spinner="dots"):
            operation = DataOperations(state['df'])
            distinct_products = operation.distinctProductIdentification(state['column_categories'])
            console.print(f"\n[light_sky_blue1]Distinct Products:[/]\n")
            theme_and_color.print_items_as_panels(distinct_products)
        return {'distinct_products': distinct_products}

    def _dataValidationNode(self, state: DataValidationState):
        with console.status(f"[plum1] Data Validation Node setting up...[/]", spinner="dots"):
            operation = DataOperations(state['df'])
            column_missed, typeChecks, duplicate_keys = operation.SchemaValidation(state['column_categories'])
            date_series_inconsistense = operation.DateSeriesConsistency(state['column_categories'])
            business_sanity_check = operation.BusinessSanitycheck(state['column_categories'])
            missing_summary_based_on_products = {}
            outlier_summary_based_on_products = {}
            for product_id in state['distinct_products']:
                missing_summary_based_on_products[product_id] = operation.MissingValueCheck(state['column_categories'], product_id)
                outlier_summary_based_on_products[product_id] = operation.OutlierDetection(state['column_categories'], product_id)

            data_val_report = f"""\n
            # Data Validation Summary

            ## Schema Validation
            - **Columns Missed**: {column_missed}
            - **Type Checks**: {typeChecks}
            - **Duplicate Keys**: {duplicate_keys}

            ## Date Series Consistency
            - **Inconsistencies**: {date_series_inconsistense}

            ## Business Sanity Check
            - **Results**: {business_sanity_check}

            ## Missing Value Summary by Product
            {"".join([f"- **Product {pid}**: {missing_summary_based_on_products[pid]}\n" for pid in missing_summary_based_on_products])}

            ## Outlier Detection Summary by Product
            {"".join([f"- **Product {pid}**: {outlier_summary_based_on_products[pid]}\n" for pid in outlier_summary_based_on_products])}
            """
            # Save to file (overwrite if exists)
            report_path = os.path.join(self.memory_path, "data_val_report.txt")
            os.makedirs(self.memory_path, exist_ok=True)  # Ensure directory exists
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(data_val_report)

            theme_and_color.print_markdown_content(data_val_report)
        console.print(f"\n[plum1]🎉 All agents finished![/]")
        return {'data_val_report': data_val_report}
    
    def _build_graph(self):
        g = StateGraph(DataValidationState)
        g.add_node("loaderNode",self._loaderNode)        
        g.add_node("colCategorizeNode",self._colCategorizeNode)        
        g.add_node("colCatApprovalNode",self._colCatApprovalNode)        
        g.add_node("distinctProductNode",self._distinctProductNode)   
        g.add_node("dataValidationNode",self._dataValidationNode)   
        
        g.add_edge(START, "loaderNode")
        g.add_edge("loaderNode", "colCategorizeNode")
        g.add_edge("colCategorizeNode", "colCatApprovalNode")
        g.add_conditional_edges(
            "colCatApprovalNode",
            self._colCatDecisionNode,
            {
                "approved": "distinctProductNode",
                "retry" : "colCategorizeNode",
                "suggested_or_denied" : "colCatApprovalNode"
            }
        )
        g.add_edge("distinctProductNode","dataValidationNode")
        g.add_edge("dataValidationNode", END)
        return g.compile(name = self.agent_name)


agent = DataValidationAgent(agent_name = "prtk", model_name = 'llama3.1', verbose=True)
agent.run()