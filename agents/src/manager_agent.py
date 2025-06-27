from agents.agent_base import AgentBase
from agents.src.data_analysis_agent import DataValidator
from typing import TypedDict
import pandas as pd

class ProjectState(TypedDict):
    df: pd.DataFrame

class ManagerAgent(AgentBase):
    def __init__(self, client, logger):
        super().__init__(client)
        self.logger = logger
        self.data_analysis_agent = DataValidator(client)
    
    def run(self, df:pd.DataFrame)
     issues, checks = self.data_analysis_agent.analyze(user_file_path, user_prompt)