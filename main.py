

from ollama import Client
from tools.tools_data_analysis import execute_python_code_on_df,data_describe,duplicate_checker, add_two_numbers, age_fn, multiply_two_numbers
import pandas as pd
from utils.memory_handler import DataStore
from utils import utility
from utils import theme_utility
import numpy as np
from rich import print


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

CLIENT = Client()
MODEL = "llama3.1"


theme_utility.print_logo()

# tools = [data_describe, duplicate_checker, sum_tool()]



# async def main():
    # agent = DataValidationAgent(
    #     agent_name = "data validator",
    #     agent_description="",
    #     tools = [],
    #     log_path = "logs/data_validaiton.log",
    #     model = "llama3-groq-tool-use:8b",
    #     df = pd.read_excel(r"C:\Users\nigam\Documents\AutoMMM\data_to_model.xlsx",sheet_name = "Sheet1")
    # )
    # resp = await agent.graph_invoke(query= "run the data validaiton")
    # print(resp)
    
    # TODO manage suggestions and loop
    # agent = ToolAgent(
    #     agent_name="Data validator",
    #     agent_description="good at data analytics",
    #     agent_goal="validate the data",
    #     tools=[add_two_numbers,
    #             generate_validation_summary,
    #             age_fn,
    #             validate_column_name_format,
    #             validate_date_format,
    #             multiply_two_numbers,
    #             duplicate_checker,
    #             validate_time_granularity,
    #             raise_validation_warnings],
    #     model="llama3-groq-tool-use:8b"
    # )
    # query = "add 3 in raj age, multiply krishna's age with the answer a
    # nd how old is pranay?\n"
    # query += "use age_fn to extract the age\n"
    # query += "sum_tool to add two numbers and multiply_two_numbers to multiply two numbers"
    # resp  = agent.graph_invoke(query = "validate the data")
    # print(resp)

    
    # from agent_patterns.react_agent.react_agentbkp import ReactAgent
    # # inputs = "add 3 in raj age, multiply krishna's age with the answer and how old is pranay?"
    # DataStore.set_df("master_data", pd.read_excel(r"C:\Users\nigam\Documents\AutoMMM\data_to_model.xlsx", sheet_name="Sheet1"))
    # DataStore.set_df("column_config_df", pd.read_excel(r"C:\Users\nigam\Documents\AutoMMM\user_inputs\column_config.xlsx"))
    # inputs = "run the data validaiton"
    # inputs = "Hi, My name is prateek, what is krishna's age"
    # # printst(inputs)
    # agent = ReactAgent(
    #     agent_name="Data validator",
    #     tools=[add_two_numbers,
    #             generate_validation_summary,
    #             age_fn,
    #             validate_column_name_format,
    #             validate_date_format,
    #             multiply_two_numbers,
    #             duplicate_checker,
    #             validate_time_granularity,
    #             raise_validation_warnings],
    #     provider="groq",
    #     backstory="You are a good at data analytics")
    # result = agent.run(inputs)



from agents.DataHandlingTeam.DataTeamManager import DataTeamManagerAgent
from agents.ModelRunnerTeam.ModellingTeamManager import ModellingTeamManagerAgent
from agent_patterns.agenticRAG.agentic_rag import AgenticRag
from agent_patterns.states import modelConfigSchema

def run_chatbot(user_input: str):
    config = {"configurable": {"thread_id": "3"}}
    agent = ModellingTeamManagerAgent(
        agent_name="Modelling Team Manager",
        agent_description=(
                "You are working with a team of market mix modelling experts as a manager"
                "Oversees the end-to-end hirarchial bayesian regression modeling workflow. "
                "Works with the configuration_architect_agent to prepare and finalize model configuration, "
                "then runs the HBR model, evaluates its performance, and, in collaboration with the user, "
                "decides whether tuning is needed. If approved, tunes the model; otherwise, ends the process."
            ),
        backstory="""
You coordinate a team of specialized agents to manage the regression modeling process:
    - configuration_architect_agent: set up, Prepares and finalizes the model configuration and model runner configurations.
    - runner_agent: Executes the model using the current configuration.
    - model_evaluator_agent: Reviews model results, checking for stability and overfitting.
    - tuner_agent: Updates configuration to improve performance when necessary.
You guide the workflow from setup to evaluation, making tuning decisions in collaboration with the user.
""".strip())

    state = {
        'messages':[{"role": "user", "content": ""}],
        'meta_model_config': {},
        'model_config': {
            'kpi': ['intercept'],
            'prior_mean': [0],
            'prior_sd': [100],
            'is_random': [1],
            'lower_bound': [np.nan],
            'upper_bound': [np.nan],
            'compute_contribution': [0]
        }
    }

    result = agent.graph.invoke(state, config)
    return result
    # rag_agent = AgenticRag(memory_folder_path = "memory")
    # for chunk in rag_agent.graph.stream({"messages": [{"role": "user","content": user_input}]}):
    #     for node, update in chunk.items():
    #         print("Update from node", node)
    #         print(update)
    #         print("\n\n")

    # return None

async def main():
    state = None
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        state = run_chatbot(user_input)
        print("Bot:", state)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



# while True:
#     rag_agent = AgenticRag(memory_folder_path = "memory")
#     user_input = input("USER: ")
#     if user_input == "break":
#         break
#     else:
#         for chunk in rag_agent.graph.stream({"messages": [{"role": "user","content": user_input}]}):
#             for node, update in chunk.items():
#                 print("Update from node", node)
#                 print(update)
#                 print("\n\n")
