"""
Configuration Architect
Role: Designs, validates, and optimizes the meta and detailed configuration settings that govern model behavior.
Responsibilities:
    - Lead the interactive setup process for model meta-configuration and feature-level settings.
    - Translate analytical goals and constraints into structured, schema-compliant configurations.
    - Leverage LLM reasoning to suggest parameter adjustments, with human-in-the-loop verification.
    - Ensure configurations are consistent, complete, and aligned with business and modeling requirements.
    - Deliver finalized configurations in standardized formats (JSON, Excel) for seamless integration into the modeling pipeline.
"""


from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from itertools import zip_longest
import time
from langgraph.types import Command
from agent_patterns.states import ConfigurationArchitectState
from agent_patterns.structured_response import ColumnCategoriesResponse
from utils import utility
from utils.memory_handler import DataStore
from utils import theme_utility
from utils import chat_utility
import pandas as pd
from rich import print
from utils.theme_utility import console, log
from markdown_pdf import MarkdownPdf, Section
from pathlib import Path
from agent_patterns.states import modelConfigSchema
import win32com.client as win32
from openpyxl.utils import get_column_letter
import pandas as pd
import os
import shutil
from pydantic import BaseModel, Field
from typing import List, Literal

modellingPrompts = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "ModellingTeamManagerPrompt",
)

class ConfigurationArchitectAgent:
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        model: str,
        log_path: str = "logs/configuration_architect.log",
    ):
        self.agent_name = agent_name
        self.agent_description = agent_description

        self.model = model
        self.llm = ChatOllama(model=self.model)
        self.log_path = log_path
        theme_utility.setup_console_logging(log_path)
        self.graph = self._build_graph()

        self.DEFAULT_CONFIG = {'rng_seed': 432, 'brand': 'nuigCare', 'Segment': ['CSCARE'], 'target_variable': 'units_sold', 'master_data_filepath': 'C:\\Users\\nigam\\Documents\\AutoMMM\\synthetic_data_generation\\synthetic_data.xlsx', 'master_data_sheet_name': 'Sheet1', 'intercept_column': '', 'configuration_filepath': 'C:\\Users\\nigam\\Documents\\AutoMMM\\user_inputs\\model_config.xlsx', 'output_dir': 'C:\\Users\\nigam\\Documents\\AutoMMM\\output', 'key_cols': ['brand', 'sku', 'date', 'Segment'], 'is_train_test_flag_already_there': False, 'perc_test': 0.2, 'draws': 4000, 'tune': 1000, 'run_correlation_vif_calculation': True, 'run_contributions': True, 'run_rowwise': False, 'itr_name': 'itr1', 'rank_variables': [], 'rank_threshold': 0.02}

    def supervisor(self, state: ConfigurationArchitectState):
        sysPrompt = "Based on the input messages, choose the most appropriate module. For tasks involving preparation of model meta data configuration, proceed to meta_model_configuration; otherwise, route to model_config_manager if json is provided or tuning changes are mentioned"
        class SupRouter(BaseModel):
            decision_node : Literal['model_config_manager','meta_config_manager']
        messages = [chat_utility.build_message_structure(role = "system", message = sysPrompt)]
        messages += state['messages']
        response = self.llm.with_structured_output(SupRouter).invoke(messages)
        approved, suggested_node = chat_utility.ask_user_approval(f"moving to [green3]{response.decision_node}[/]")
        if approved is True:
            return Command(goto = response.decision_node)
        else:
            while suggested_node not in ['model_config_manager','meta_config_manager']:
                suggested_node = chat_utility.take_user_input(
                    "Invalid mode name. Please re-enter one of ['model_config_manager','meta_config_manager']"
                ).strip()
            return Command(goto = suggested_node)

    def meta_config_manager(self, state: ConfigurationArchitectState):
        log("[medium_purple3]LOG: Starting model meta Configuration Setup...[/]")
        with console.status(f"[plum1] model meta Configuration Setting up...[/]", spinner="dots"):
            console.print("")

        final_config = self.DEFAULT_CONFIG.copy()
        sysprompt = modellingPrompts['metaConfigManagerPrompt'].format(
                conf = self.DEFAULT_CONFIG
            ).strip()
        state['messages'] += [chat_utility.build_message_structure(role = "system", message = sysprompt)]
        
        take_input_prompt = f"Hi, let's set up the model's meta configuration. Press ⏎ to continue"        
        feedback_message = None
        while True:
            if feedback_message is None:            
                user_input = chat_utility.take_user_input(take_input_prompt)
            else:
                user_input = feedback_message
                feedback_message = None

            if user_input.lower().strip() == "exit":
                break

            state['messages'] += [chat_utility.build_message_structure(role="user", message=user_input)]
            with console.status(f"[plum1] Generating response...[/]\n", spinner="dots"):
                response = self.llm.invoke(state['messages'])
            
            parsed_dict = chat_utility.parse_json_from_response(response.content)
            if parsed_dict and isinstance(parsed_dict, dict):
                for key, value in parsed_dict.items():
                    if key in final_config:
                        final_config[key] = value

                if all(k in final_config for k in self.DEFAULT_CONFIG):
                    theme_utility.print_dictionary(final_config, title = "Model Meta Configuration")
                    approved, message = chat_utility.ask_user_approval("Final Config")

                    if approved is True:
                        log("[green3]LOG: Final meta config approved by user. [/]")
                        return Command(
                            goto = "model_config_manager",
                            update = {
                                "messages": [
                                    chat_utility.build_message_structure(role = "assistant", message = f"meta_model_config: \n{final_config}")
                                    ],
                                'meta_model_config': final_config
                                }
                            )
                    else:
                        feedback_message = message
                        chat_utility.append_to_structure(state['messages'], role="user", message=message)
            else:
                theme_utility.display_response(response.content, title=self.agent_name)
                chat_utility.append_to_structure(state['messages'], role="assistant", message=response.content)
                
            take_input_prompt = "USER"    
        
    def model_config_manager(self, state: ConfigurationArchitectState):
        log("[medium_purple3]LOG: Starting model configuration setup...[/]")
        with console.status(f"[plum1]Setting up model configuration...[/]", spinner="dots"):
            console.print("")
        try:
            actual_cols = list(DataStore.get_df("master_data").columns)
        except:            
            synthetic_filepath = chat_utility.take_user_input("Provide synthetic_data file path")
            while not os.path.isfile(synthetic_filepath):
                synthetic_filepath = chat_utility.take_user_input(
                    "Invalid path. Please re-enter Excel file path"
                ).strip()
            excel_file = pd.ExcelFile(synthetic_filepath)
            console.print("[sandy_brown]sheet_names[/] ", excel_file.sheet_names)
            sheet_name = chat_utility.take_user_input("[sandy_brown]Sheet[/] ").strip()
            while sheet_name not in excel_file.sheet_names:
                sheet_name = chat_utility.take_user_input(
                    "Invalid sheet name. Please re-enter "
                ).strip()
            actual_cols = list(pd.read_excel(synthetic_filepath, sheet_name = sheet_name).columns)

        config_ = state['model_config']

        model_config_df = pd.DataFrame({
            'kpi' : config_['kpi'],
            'prior_mean' : config_['prior_mean'],
            'prior_sd' : config_['prior_sd'],
            'is_random' : config_['is_random'],
            'lower_bound' : config_['lower_bound'],
            'upper_bound' : config_['upper_bound'],
            'compute_contribution' : config_['compute_contribution']
        })
        log(
            f"[medium_purple3] LOG: Initial model_config_df created with[/] - [turquoise4]{len(model_config_df)} rows[/]"
        )

        col_config = ["kpi", "prior_mean", "prior_sd", "is_random", "lower_bound", "upper_bound", "compute_contribution"]
        rows = list(zip_longest(model_config_df.values.tolist(), actual_cols, fillvalue=None))
        merged_data = []
        for kpi_row, col_name in rows:
            if kpi_row is None:
                kpi_row = [None] * len(col_config)
            merged_data.append(list(kpi_row) + [col_name])

        required_columns_df = pd.DataFrame(merged_data, columns=col_config + ["all_columns"])
        log("[medium_purple3]LOG: Merged KPI rows with actual columns[/]")

        file_path = r"user_inputs\model_config.xlsx"
        chat_utility.user_input_excel(required_columns_df, file_path=file_path)
        
        model_config_df = pd.read_excel(file_path)
        
        transformation_sheet = pd.DataFrame({
                "kpi": list(model_config_df['kpi']),
                "take_log": 0,
                "scale_min_max_asin": 0,
                "scale_min_max_brand": 0,
                "normalize_distribution": 0
            })
        
        adstock_sheet = pd.DataFrame(columns=[
                "kpi",
                "sku",
                "adstock_rate",
                "s_curve_param_b",
                "s_curve_param_c"
            ])
        
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            model_config_df.to_excel(writer, sheet_name="input_features", index=False)
            transformation_sheet.to_excel(writer, sheet_name="general_transformations", index=False)
            adstock_sheet.to_excel(writer, sheet_name="adstock_s_curve_transformations", index=False)
        log(
            f"[green3] LOG: Written configuration, transformation, and adstock sheets to[/] - [turquoise4]{file_path}[/]"
        )
        chat_utility.append_to_structure(state['messages'], role="assistant", message=f"Written configuration, transformation, and adstock sheets {file_path}")
        console.print(f"\n[light_sky_blue1]Columns detected: [/]")
        theme_utility.print_items_as_panels(list(model_config_df['kpi']))
        
        config_dict = {col: model_config_df[col].tolist() for col in model_config_df.columns}

        model_config_schema = modelConfigSchema(**config_dict)
        log("[green3]LOG: Instantiated modelConfigSchema successfully[/]")
        return {'model_config': model_config_schema}
    
    def _build_graph(self):
        g = StateGraph(ConfigurationArchitectState)
        g.add_node("supervisor", self.supervisor)
        g.add_node("meta_config_manager", self.meta_config_manager)
        g.add_node("model_config_manager", self.model_config_manager)

        g.add_edge(START, "supervisor")
        g.add_edge("model_config_manager", END)
        return g.compile(name=self.agent_name)