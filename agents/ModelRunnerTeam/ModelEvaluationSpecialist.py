"""
Model Evaluation Specialist
Role: Evaluates regression model outputs and provides insights for tuning decisions.
Responsibilities:
    - Analyze model performance metrics (R², RMSE, coefficient stability, contribution accuracy).
    - Interpret configuration and coefficient outputs to assess model reliability.
    - Recommend tuning adjustments based on coefficient credibility and performance analysis.
    - Combine insights into a final evaluation report for decision-making and further action.
"""

from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from itertools import zip_longest
import time
from langgraph.types import Command
from agent_patterns.states import ModelEvaluatorState
from agent_patterns.structured_response import ColumnCategoriesResponse
from utils import utility
from utils.memory_handler import DataStore
from utils import theme_utility
from utils import chat_utility
import pandas as pd
import json
from rich import print
from utils.theme_utility import console, log
from markdown_pdf import MarkdownPdf, Section
from pathlib import Path
from agent_patterns.states import modelConfigSchema
import win32com.client as win32
from openpyxl.utils import get_column_letter
import pandas as pd
import os
import glob


evaluationPrompts = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "ModellingTeamManagerPrompt",
)

class ModelEvaluatorAgent:
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        model: str,
        log_path: str = "logs/model_evaluator.log",
    ):
        self.agent_name = f"{agent_name}: Abhijeet"
        self.agent_description = agent_description

        self.model = model
        self.llm = ChatOllama(model=self.model)
        self.log_path = log_path
        theme_utility.setup_console_logging(log_path)
        self.graph = self._build_graph()


    def evaluator_agent_prep(self, state: ModelEvaluatorState):
        log("[medium_purple3]LOG: Evaluator agent Configuration Setup...[/]")
        with console.status(f"[plum1] Evaluator agent Setting up...[/]", spinner="dots"):
            console.print("")
        config = dict(state['model_config'])
        meta_model_config = state['meta_model_config']
        folder_path = os.path.join(meta_model_config['output_dir'],f"Iteration_{meta_model_config['itr_name']}")
        for idx, file in enumerate(glob.glob(f"{folder_path}/*.xls*"), start=1):
            print(f"{idx}. {file.split('/')[-1]}")

        choice = int(chat_utility.take_user_input("Enter the number of the file you want to select"))
        if 1 <= choice <= len(glob.glob(f"{folder_path}/*.xls*")):
            selected_file = glob.glob(f"{folder_path}/*.xls*")[choice - 1]

        log(
            f"[medium_purple3] LOG: Selected file: [/] - [turquoise4]{selected_file}[/]"
        )
        performance = pd.read_excel(selected_file, sheet_name = "evaluation_metrics")
        coef = pd.read_excel(selected_file, sheet_name = "model_coefficients")
        mapping = pd.read_excel(selected_file, sheet_name = "sku_to_code_mapping")
        code_to_sku = mapping.set_index("code")["sku"].to_dict()
        coef["index"] = coef["index"].str.replace(
                r"\[(\d+)\]",
                lambda m: f"[{code_to_sku[int(m.group(1))]}]",
                regex=True
            )
        self.config = config
        self.performance = performance

        coef = coef[['index', 'mean', 'sd', 'hdi_3%', 'hdi_97%','r_hat','auc < 0']]
        coef['hdi_width'] = coef['hdi_97%'] - coef['hdi_3%']
        coef['credible'] = (coef['hdi_3%'] > 0) | (coef['hdi_97%'] < 0)
        self.coef  = coef[['index', 'mean', 'sd', 'hdi_width', 'credible', 'r_hat']]
        log("[green3]LOG: Loaded matrices for evaluation[/]")
        return state
    
    def config_interpreter(self, state:ModelEvaluatorState):
        log("[medium_purple3]LOG: Interpreting model configuration...[/]")
        # with console.status(f"[plum1] Interpreting model configuration...[/]", spinner="dots"):
        sysPrompt = evaluationPrompts['ConfigInterpreterPrompt'].format(
            conf = self.config
        )
        response = self.llm.invoke(sysPrompt)
        theme_utility.display_response(response.content)
        return {'config_interpreter': response.content}
    
    def performance_analyst(self, state:ModelEvaluatorState):
        log("[medium_purple3]LOG: Analyzing model performance metrics...[/]")
        # with console.status(f"[plum1] Analyzing model performance metrics...[/]", spinner="dots"):
        sysPrompt = evaluationPrompts['PerformanceAnalystPrompt'].format(
            performance = self.performance.to_dict(orient='records')
            )
        response = self.llm.invoke(sysPrompt)
        theme_utility.display_response(response.content)
        return {'performance_analyst': response.content}
    
    def coef_explainer(self, state:ModelEvaluatorState):
        log("[medium_purple3]LOG: Explaining coefficient outputs...[/]")
        # with console.status(f"[plum1] Explaining coefficient outputs...[/]", spinner="dots"):
        coef_json = json.dumps(self.coef.to_dict(orient='records'))
        sysPrompt = evaluationPrompts['CoefExplainerPrompt'].format(
            coef = coef_json
            )
        response = self.llm.invoke(sysPrompt)
        theme_utility.display_response(response.content)
        return {'coef_explainer': response.content}
    
    def tuning_recommender(self, state:ModelEvaluatorState):
        log("[medium_purple3]LOG: Generating tuning recommendations based on coefficients...[/]")
        # with console.status(f"[plum1] Generating tuning recommendations based on coefficients...[/]", spinner="dots"):
        coef_json = json.dumps(self.coef.to_dict(orient='records'))
        sysPrompt = evaluationPrompts['TuningRecommenderPrompt'].format(
            coef = coef_json
            )
        response = self.llm.invoke(sysPrompt)
        theme_utility.display_response(response.content)
        return {'tuning_recommender': response.content}

    def report_combiner(self, state):
        log("[medium_purple3]LOG: Combining evaluation insights into a final report...[/]")
        with console.status(f"[plum1] Combining evaluation insights into a final report...[/]", spinner="dots"):
            sysPrompt = evaluationPrompts['RecommenderCombinerPrompt'].format(
                config_interpreter_info = state['config_interpreter'],
                performance_analyst_info = state['performance_analyst'],
                coef_explainer_info = state['coef_explainer'],
                tuning_recommender_info = state['tuning_recommender']
                )
            response = self.llm.invoke(sysPrompt)
        theme_utility.display_response(response.content)
        log("[green3]LOG: Final report generated — analysis completed.[/]")
        return {'final_report': response.content}

    def _build_graph(self):
        g = StateGraph(ModelEvaluatorState)
        g.add_node("evaluator_agent_prep", self.evaluator_agent_prep)
        g.add_node("config_interpreter", self.config_interpreter)
        g.add_node("performance_analyst", self.performance_analyst)
        g.add_node("coef_explainer", self.coef_explainer)
        g.add_node("tuning_recommender", self.tuning_recommender)
        g.add_node("report_combiner", self.report_combiner)

        g.add_edge(START, "evaluator_agent_prep")
        g.add_edge("evaluator_agent_prep", "config_interpreter")
        g.add_edge("evaluator_agent_prep", "performance_analyst")
        g.add_edge("evaluator_agent_prep", "coef_explainer")
        g.add_edge("evaluator_agent_prep", "tuning_recommender")

        g.add_edge("config_interpreter", "report_combiner")
        g.add_edge("performance_analyst", "report_combiner")
        g.add_edge("coef_explainer", "report_combiner")
        g.add_edge("tuning_recommender", "report_combiner")

        g.add_edge("report_combiner", END)
        return g.compile(name=self.agent_name)