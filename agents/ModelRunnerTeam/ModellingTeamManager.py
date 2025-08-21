from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pandas.io.pytables import performance_doc
from agent_patterns.states import ModellingTeamManagerState
from agent_patterns.states import modelConfigSchema
from utils import chat_utility
from utils import theme_utility
from utils.memory_handler import DataStore

from langchain_ollama import ChatOllama
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from utils.theme_utility import console
from itertools import zip_longest
from langgraph.types import Command
from langchain_core.prompts.chat import ChatPromptTemplate
import tempfile
import shutil
import glob
import numpy as np
import os
from agents.ModelRunnerTeam.ModelExecutionSpecialist import RunnerAgent
from agents.ModelRunnerTeam.ModelEvaluationSpecialist import ModelEvaluatorAgent
from agents.ModelRunnerTeam.configuration_architect import ConfigurationArchitectAgent
from agents.DataHandlingTeam.DataQualityAnalyst import DataQualityAnalystAgent
import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

import json
import ast


from rich import print
import pprint
from utils import utility

ManagerPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "ModellingTeamManagerPrompt",
)

user_messages = utility.load_prompt_config(
    r"prompts\user_messages.yaml",
    "data_team_manager",
)


class ModellingTeamManagerAgent:
    def __init__(self, agent_name :str, agent_description:str, backstory:str = ""):
        self.agent_name = f"{agent_name}: Jeevan"
        self.agent_description = agent_description
        self.graph = self.build_graph(ModellingTeamManagerState)
        self.backstory = backstory
        self.llm = ChatOllama(model = "llama3.1")
        self.openaillm = ChatOpenAI(model = "gpt-4o-mini")
        self.configuration_architect = ConfigurationArchitectAgent(
            agent_name="configuration Architect Agent",
            agent_description="Designs and manages model configuration, transformations, and readiness for Market Mix Modelling workflows.",
            model="llama3.1")
        self.hbr_runner_agent = RunnerAgent(
            agent_name="HBR Model Execution Specialist Agent",
            agent_description="Executes the regression model and produces outputs for evaluation.",
            config_path='user_inputs/config.py',
            runner_path='C:/Users/nigam/Documents/hbr-numpyro/runner.py')
        self.model_evaluator_agent = ModelEvaluatorAgent(
            agent_name="Model Evaluator Agent",
            agent_description="Assesses regression model outputs, interprets configuration, analyzes performance metrics, explains coefficients, and provides tuning recommendations for Marketing Mix Modeling workflows.",
            model="llama3.1"
            )       

        
    def manager_agent_node(self, state: ModellingTeamManagerState):
        sysPrompt = ManagerPrompt['introdutionPrompt'].format(
            agent_name = self.agent_name,
            agent_description = self.agent_description,
            agent_backstory = self.backstory,
            history = state['messages']
        )
        with console.status(f"[plum1] Generating response from {self.agent_name}...", spinner="dots"):
            response = self.llm.invoke(sysPrompt)        
        theme_utility.display_response(response.content)

        prompt = ManagerPrompt['manager_chat_prompt'].format(
            agent_name = self.agent_name,
            agent_description = self.agent_description,
            backstory = self.backstory
        )
        
        state['messages'] = state['messages'] + [
                    chat_utility.build_message_structure(role = "system", message = prompt)
                ]
                
        take_input_prompt = f"Hello! I'm {self.agent_name}. Ready to help—what would you like to do first?"
        while True:            
            user_input = chat_utility.take_user_input(take_input_prompt)
            state['messages'] = state['messages'] + [
                    chat_utility.build_message_structure(role = "user", message = user_input)
                ]
            take_input_prompt = "USER"
            if user_input == "exit":
                break
            response = self.llm.invoke(state['messages'])
            theme_utility.display_response(response.content, title = self.agent_name)
            parsed = chat_utility.parse_json_from_response(response.content)
            if parsed and isinstance(parsed, dict) and "call_agent" in parsed:
                approved, feedback = chat_utility.ask_user_approval(agent_name = parsed['call_agent']) 
                if approved is True:
                    response = f"Executing {parsed['call_agent']} with input: {parsed.get('task', '')}"
                    messages = state['messages'] + [
                        chat_utility.build_message_structure(role = "assistant", message = response)
                    ]
                    return Command(
                        goto = parsed['call_agent'],
                        update = {
                            "next_agent": parsed['call_agent'],
                            "task": parsed.get("task", ""),
                            "messages" : messages
                            }
                    )
                else:
                    response = f"user denied to move to {parsed['call_agent']}, with this input : {feedback}"
                    chat_utility.append_to_structure(state['messages'], role="assistant", message = response)
            else:
                chat_utility.append_to_structure(state['messages'], role="assistant", message = response.content)

    def configuration_architect_node(self, state: ModellingTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.configuration_architect.agent_name,
            agent_description=self.configuration_architect.agent_description,
            is_interactive=True,
        )
        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=state['task'])
            ],
            'meta_model_config': state['meta_model_config'],
            'model_config': state['model_config']
        }
        ca_response = self.configuration_architect.graph.invoke(inputs)
        messages_from_configuration_architect = ca_response['messages']
        sysprompt = f"You are {self.configuration_architect.agent_name},{self.configuration_architect.agent_description} , working for a {self.agent_name}, based on you history of messages given by user, reply back to him on completion or status of your allotted task."
        messages = [
            chat_utility.build_message_structure(role = "system", message = sysprompt),
            chat_utility.build_message_structure(role = "user", message = json.dumps(messages_from_configuration_architect))
        ]
        response = self.llm.invoke(messages)
        theme_utility.display_response(response.content, title = self.configuration_architect.agent_name)
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.configuration_architect.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "manager",
                update = {
                    "next_agent": "manager",
                    "messages": [chat_utility.build_message_structure(role = "assistant", message = response.content)],
                    'model_config': ca_response['model_config'],
                    'meta_model_config': ca_response['meta_model_config']
                    }
            )
        else:
            return Command(
                goto = "manager",
                update = {
                    "next_agent": "manager",
                    "task": feedback,
                    "messages": [
                        chat_utility.build_message_structure(role = "assistant", message = response.content),
                        chat_utility.build_message_structure(role = "user", message = feedback)
                        ],
                    'model_config': ca_response['model_config'],
                    'meta_model_config': ca_response['meta_model_config']
                    }
            )
    
    def runner_agent(self, state: ModellingTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.hbr_runner_agent.agent_name,
            agent_description=self.hbr_runner_agent.agent_description,
            is_interactive=True,
        )
        with open("user_inputs/config.py", "w", encoding="utf-8") as f:
            f.write("# Auto-generated configuration file\n\n")
            f.write("meta_data_config = " + repr(state['meta_model_config']) + "\n")
        success, error = self.hbr_runner_agent.run()
        if success is False:
            return Command(
                goto="manager", 
                update={
                    "messages": [
                        chat_utility.build_message_structure(role = "assistant", message = f"model running failed with error :{error}")
                        ]
                    }
                )
        state['messages'].append({"role": "assistant", "content": f"{self.hbr_runner_agent.agent_name} completed running the hbr model: {success}"})
        os.startfile(Path(state['meta_model_config']['output_dir']))
        return Command(goto="model_evaluator_agent", update=state)
    
    def model_evaluator_node(self, state: ModellingTeamManagerState):
        theme_utility.print_startup_info(
            agent_name=self.model_evaluator_agent.agent_name,
            agent_description=self.model_evaluator_agent.agent_description,
            is_interactive=True,
        )
        inputs = {
            "messages": [
                chat_utility.build_message_structure(role="user", message=state['task'])
            ],
            'meta_model_config': state['meta_model_config'],
            'model_config': state['model_config']
        }
        evaluator_agent_response = self.model_evaluator_agent.graph.invoke(inputs)
        messages_from_configuration_architect = evaluator_agent_response['final_report']
        sysprompt = f"You are {self.configuration_architect.agent_name},{self.configuration_architect.agent_description} , working for a {self.agent_name}, based on you history of messages given by user, reply back to him on completion or status of your allotted task."
        messages = [
            chat_utility.build_message_structure(role = "system", message = sysprompt),
            chat_utility.build_message_structure(role = "user", message = json.dumps(messages_from_configuration_architect))
        ]
        response = self.llm.invoke(messages)
        theme_utility.display_response(response.content, title = self.configuration_architect.agent_name)
        approved, feedback = chat_utility.ask_user_approval(agent_name = self.configuration_architect.agent_name, prompt_suffix="Use this input, or modify before sending to manager agent?") 
        if approved is True:
            return Command(
                goto = "manager",
                update = {
                    "next_agent": "manager",
                    "messages": [chat_utility.build_message_structure(role = "assistant", message = response.content)],
                    "config_interpreter" : evaluator_agent_response['config_interpreter'],
                    "performance_analyst" : evaluator_agent_response['performance_analyst'],
                    "coef_explainer" : evaluator_agent_response['coef_explainer'],
                    "tuning_recommender" : evaluator_agent_response['tuning_recommender'],
                    "final_report" : evaluator_agent_response['final_report']
                    }
            )
        else:
            return Command(
                goto = "manager",
                update = {
                    "next_agent": "manager",
                    "task": feedback,
                    "messages": [
                        chat_utility.build_message_structure(role = "assistant", message = response.content),
                        chat_utility.build_message_structure(role = "user", message = feedback)
                        ],
                    "config_interpreter" : evaluator_agent_response['config_interpreter'],
                    "performance_analyst" : evaluator_agent_response['performance_analyst'],
                    "coef_explainer" : evaluator_agent_response['coef_explainer'],
                    "tuning_recommender" : evaluator_agent_response['tuning_recommender'],
                    "final_report" : evaluator_agent_response['final_report']
                    }
            )

    def model_tuner_node(self, state: ModellingTeamManagerState):
        current_model_config = dict(state['model_config'])
        tuner_conversation = []
        # memory = DataStore.get_str("memory_context")
        with open(r"memory\memory_context.txt", "r", encoding="utf-8") as f:
            memory = f.read()

        sysPrompt = ManagerPrompt['BrainStoringPrompt'].format(
            memory = memory,
            current_config = current_model_config
        )

        tuner_conversation += [chat_utility.build_message_structure(role = "system", message = sysPrompt)]        
        
        take_input_prompt = "USER"
        feedback_message = None
        while True:
            if feedback_message is None:            
                user_input = chat_utility.take_user_input(take_input_prompt, default = "done")
            else:
                user_input = feedback_message
                feedback_message = None

            if user_input.lower().strip() == "exit":
                break

            tuner_conversation += [chat_utility.build_message_structure(role="user", message=user_input)]

            response = self.openaillm.invoke(tuner_conversation)       

            if user_input.lower().strip() == "done":
                approval, _ = chat_utility.ask_user_approval("discussion board", "Do you want tuning?")
                if approval is True:
                    file_path = r"user_inputs\model_config.xlsx"
                    model_config_df = pd.read_excel(file_path, sheet_name = 'input_features')
                    chat_utility.user_input_excel(model_config_df, file_path=r"user_inputs\temp.xlsx")
                    
                    updated_model_config = pd.read_excel(r"user_inputs\temp.xlsx")
                    with pd.ExcelWriter(file_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
                        updated_model_config.to_excel(writer, sheet_name="input_features", index=False)
                    os.remove(r"user_inputs\temp.xlsx")
                    return Command(goto = "hbr_runner_agent")
                else:
                    return Command(goto = "manager", update = {"messages", [tuner_conversation]})

            else:
                theme_utility.display_response(response.content, title=self.agent_name)
                chat_utility.append_to_structure(tuner_conversation, role="assistant", message=response.content)

            take_input_prompt = "USER"                           


    def build_graph(self, state:ModellingTeamManagerState):
        workflow = StateGraph(state)
        workflow.add_node("manager", self.manager_agent_node)
        workflow.add_node("configuration_architect_agent", self.configuration_architect_node)
        
        workflow.add_node("hbr_runner_agent", self.runner_agent)
        workflow.add_node("model_evaluator_agent", self.model_evaluator_node)
        workflow.add_node("model_tuner_agent", self.model_tuner_node)

        # workflow.add_node("evaluator_agent_prep", self.evaluator_agent_prep)        
        # workflow.add_node("config_interpreter", self.config_interpreter)
        # workflow.add_node("performance_analyst", self.performance_analyst)
        # workflow.add_node("coef_explainer", self.coef_explainer)
        # workflow.add_node("tuning_recommender", self.tuning_recommender)
        # workflow.add_node("report_combiner", self.report_combiner)
        # workflow.add_node("tuner_agent", self.tuner_agent)

        workflow.add_edge(START, "manager")
        # workflow.add_edge("meta_config_manager", "model_config_manager")
        # workflow.add_edge('model_config_manager', "evaluator_agent_prep")

        # workflow.add_edge('evaluator_agent_prep', "config_interpreter")
        # workflow.add_edge('evaluator_agent_prep', "performance_analyst")
        # workflow.add_edge('evaluator_agent_prep', "coef_explainer")
        # workflow.add_edge('evaluator_agent_prep', "tuning_recommender")

        # workflow.add_edge("config_interpreter", "report_combiner")
        # workflow.add_edge("performance_analyst", "report_combiner")
        # workflow.add_edge("coef_explainer", "report_combiner")
        # workflow.add_edge("tuning_recommender", "report_combiner")
        # workflow.add_edge("report_combiner", "tuner_agent")

        # workflow.add_edge("report_combiner", END)

        return workflow.compile(checkpointer= checkpointer)