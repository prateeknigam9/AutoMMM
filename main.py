from agents.data_analysis_agent import DataValidationAgent
from agent_patterns.structured_response import TypeValidationResponse
from agent_patterns.tool_calling_agent.tool_agent import ToolAgent
import config
from ollama import Client
from tools.tools_data_analysis import execute_python_code_on_df,data_describe,duplicate_checker, sum_tool
import pandas as pd
from utils.memory_handler import DataStore
from utils import utility
from utils import theme_utility

DataValidationPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "DataValidationPrompt",
)


CLIENT = Client()
MODEL = "llama3.1"


theme_utility.print_logo()

tools = [data_describe, duplicate_checker, sum_tool()]

async def main():
    agent = DataValidationAgent(
        agent_name = "data validator",
        agent_description="",
        tools = [],
        log_path = "logs/data_validaiton.log",
        model = "llama3.1",
        df = pd.read_excel(r"C:\Users\nigam\Documents\agents\data_to_model.xlsx",sheet_name = "Sheet1")
    )
    resp = await agent.graph_invoke(query= "run the data validaiton")
    print(resp)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


# approval_msg = """
# I am about to use the tool **{tool}** to perform the task:  
# **"{task}"**,  
# because: _{reason}_.

# The tool will be called with the following arguments:  
# `{args}`

# Do you approve proceeding with this action?
# Please respond with "Yes" to continue or "No" to cancel.
# """