
import config
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
from agents.ContributionTeam.ContributionTeamManager import ContributionTeamManagerAgent
from agents.CEO.CEOAgent import CEOAgent
from agent_patterns.agenticRAG.agentic_rag import AgenticRag
from agent_patterns.states import modelConfigSchema

def run_chatbot(user_input: str):
    config = {"configurable": {"thread_id": "3"}}
    
    # CEO Agent - Main entry point for complete AutoMMM workflow
    agent = CEOAgent(
        agent_name="AutoMMM CEO",
        agent_description=(
            "You are the Chief Executive Officer of AutoMMM, overseeing the complete end-to-end workflow "
            "from data preparation to final insights. You orchestrate three specialized teams: "
            "Data Handling Team (data loading, analysis, quality assurance), "
            "Model Runner Team (HBR model configuration, execution, evaluation), and "
            "Contribution Team (marketing contribution analysis and business insights). "
            "You ensure seamless handoffs between teams, monitor project progress and quality, "
            "and provide executive-level oversight and strategic direction."
        ),
        backstory="""
You are the visionary leader of AutoMMM, a comprehensive Marketing Mix Modeling platform. 
Your role is to orchestrate the complete workflow from raw data to actionable business insights.

**Your Teams:**
- **Data Handling Team**: Led by Gaurav, manages data loading, analysis, and quality assurance
- **Model Runner Team**: Led by Jeevan, handles model configuration, execution, and evaluation  
- **Contribution Team**: Led by specialized analysts, provides marketing attribution and ROI insights

**Your Responsibilities:**
- Ensure seamless workflow execution across all three teams
- Monitor project progress, quality, and deliverables
- Make strategic decisions on project direction and optimization
- Provide executive-level insights and recommendations
- Coordinate handoffs between teams and phases
- Generate comprehensive project overviews and executive summaries

You guide the entire AutoMMM process from data preparation through model execution to final business insights, 
ensuring that each phase builds upon the previous one and delivers maximum value to stakeholders.
""".strip())

    # Initialize CEO state for complete workflow management
    state = {
        'messages': [{"role": "user", "content": user_input}],
        'data_team_status': False,
        'modelling_team_status': False,
        'contribution_team_status': False,
        'overall_project_status': 'Not Started',
        'current_phase': 'Initialization',
        'next_phase': 'Data Preparation',
        'data_team_report': {},
        'modelling_team_report': {},
        'contribution_team_report': {},
        'final_executive_summary': '',
        'task': user_input,
        'next_team': '',
        'command': None
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
#             print("\n\n")
